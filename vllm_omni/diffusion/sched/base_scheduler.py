# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from dataclasses import fields
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionBatchingConfig, OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.batching_policy import (
    DiffusionBatchingPolicy,
    create_diffusion_batching_policy,
)
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionRequestState,
    DiffusionRequestStatus,
    DiffusionSchedulerOutput,
    NewRequestData,
    RequestBatchSamplingParamsKey,
    SamplingParamsKey,
    SchedulerInterface,
)

logger = init_logger(__name__)

# LoRA identity is derived from `sampling.lora_request`, not a same-named field
# on sampling params, so it must be resolved separately from the bulk lookup.
_SAMPLING_PARAMS_KEY_FIELD_NAMES = frozenset(f.name for f in fields(SamplingParamsKey)) - {"lora_int_id"}
_REQUEST_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES = frozenset(f.name for f in fields(RequestBatchSamplingParamsKey)) - {
    "lora_int_id"
}


def get_sampling_params_key(request: OmniDiffusionRequest) -> SamplingParamsKey:
    """Build a batch-compatibility key from the request's sampling params."""
    sampling = request.sampling_params
    lora_request = getattr(sampling, "lora_request", None)
    return SamplingParamsKey(
        lora_int_id=lora_request.lora_int_id if lora_request is not None else None,
        **{name: getattr(sampling, name) for name in _SAMPLING_PARAMS_KEY_FIELD_NAMES},
    )


def get_request_batch_sampling_params_key(request: OmniDiffusionRequest) -> RequestBatchSamplingParamsKey:
    """Build a request-batch compatibility key from the request's sampling params."""
    sampling = request.sampling_params
    lora_request = getattr(sampling, "lora_request", None)
    key_kwargs = {name: getattr(sampling, name) for name in _REQUEST_BATCH_SAMPLING_PARAMS_KEY_FIELD_NAMES}
    key_kwargs["lora_int_id"] = lora_request.lora_int_id if lora_request is not None else None
    return RequestBatchSamplingParamsKey(**key_kwargs)


class _BaseScheduler(SchedulerInterface):
    """Shared queue/state bookkeeping for diffusion schedulers."""

    def __init__(self) -> None:
        self.od_config: OmniDiffusionConfig | None = None
        self._request_states: dict[str, DiffusionRequestState] = {}
        self._step_id: int = 0
        self._waiting: deque[str] = deque()
        self._waiting_by_key: dict[SamplingParamsKey | None, deque[str]] = {}
        self._running: list[str] = []
        self._running_sampling_params_key: SamplingParamsKey | RequestBatchSamplingParamsKey | None = None
        self._finished_req_ids: set[str] = set()
        self._key_arrivals: dict[SamplingParamsKey | None, int] = {}
        self._arrival_window: deque[SamplingParamsKey | None] = deque()
        self._key_deficits: dict[SamplingParamsKey | None, float] = {}
        self._drr_started: bool = False
        self._drr_cursor_index: int = 0
        self.max_num_running_reqs: int = 1
        self._prefetch_enabled: bool = False
        self.batching_config: DiffusionBatchingConfig = DiffusionBatchingConfig()
        self.batching_policy: DiffusionBatchingPolicy = create_diffusion_batching_policy(
            self.batching_config
        )

    def initialize(self, od_config: OmniDiffusionConfig) -> None:
        self.od_config = od_config
        self._request_states.clear()
        self._step_id = 0
        self._waiting.clear()
        self._waiting_by_key.clear()
        self._running.clear()
        self._running_sampling_params_key = None
        self._finished_req_ids.clear()
        self._key_arrivals.clear()
        self._arrival_window.clear()
        self._key_deficits.clear()
        self._drr_started = False
        self._drr_cursor_index = 0
        max_num_seqs = getattr(od_config, "max_num_seqs", 1)
        try:
            self.max_num_running_reqs = max(1, int(max_num_seqs))
        except (TypeError, ValueError):
            self.max_num_running_reqs = 1
        omni_kv = getattr(od_config, "omni_kv_config", None) or {}
        self._prefetch_enabled = bool(omni_kv.get("enable_kv_async_prefetch", False))
        self.batching_config = self._resolve_batching_config(od_config)
        self.batching_policy = create_diffusion_batching_policy(self.batching_config)
        self._reset_scheduler_state()

    def add_request(self, request: OmniDiffusionRequest) -> str:
        return self._add_request_with_request_id(request.request_id, request)

    def _add_request_with_request_id(self, request_id: str, request: OmniDiffusionRequest) -> str:
        if request_id in self._request_states:
            raise ValueError(f"request_id {request_id!r} is already active.")
        state = self._make_request_state(request_id, request)
        self._request_states[request_id] = state
        self._enqueue_waiting(state)
        logger.debug("%s add_request: %s (waiting=%d)", self.__class__.__name__, request_id, len(self._waiting))
        return request_id

    def schedule(self) -> DiffusionSchedulerOutput:
        scheduled_new_reqs: list[NewRequestData] = []
        scheduled_cached_request_ids: list[str] = []
        self.batching_policy.on_schedule_start()

        # First, schedule the RUNNING request(s)
        for request_id in self._running:
            state = self._request_states.get(request_id)
            if state is not None:
                scheduled_cached_request_ids.append(request_id)

        # Second, schedule WAITING requests while capacity remains.
        while self._waiting and len(self._running) < self.max_num_running_reqs:
            state = self._peek_next_waiting_state()
            if state is None:
                break
            if not self._can_schedule_waiting(state):
                break
            if len(self._running) >= self._effective_max_running_reqs(state):
                break

            self._pop_waiting_state(state)
            was_new_request = state.status == DiffusionRequestStatus.WAITING
            if not self._running:
                self._running_sampling_params_key = state.sampling_params_key
            state.status = DiffusionRequestStatus.RUNNING
            self._running.append(state.request_id)
            if was_new_request:
                scheduled_new_reqs.append(NewRequestData.from_state(state))
            else:
                scheduled_cached_request_ids.append(state.request_id)

        # Expose the next waiting request (serial mode) so the runner can
        # prefetch its KV during this forward.  Skip a request without
        # kv_sender_info (would target the wrong sender under multi-replica) or
        # one already finished/aborted (would consume its sender buffer for
        # nothing).
        kv_prefetch_jobs: dict | None = None
        if self._prefetch_enabled and self._waiting:
            nxt = self._request_states.get(self._waiting[0])
            if nxt is not None and not nxt.is_finished():
                sender_info = getattr(nxt.req, "kv_sender_info", None)
                if sender_info:
                    kv_prefetch_jobs = {
                        "request_id": nxt.request_id,
                        "kv_sender_info": sender_info,
                    }

        scheduler_output = DiffusionSchedulerOutput(
            step_id=self._step_id,
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_cached_reqs=CachedRequestData(request_ids=scheduled_cached_request_ids),
            finished_req_ids=set(self._finished_req_ids),
            num_running_reqs=len(self._running),
            num_waiting_reqs=len(self._waiting),
            kv_prefetch_jobs=kv_prefetch_jobs,
        )

        # update after schedule
        self._step_id += 1
        self._finished_req_ids.clear()
        return scheduler_output

    def has_requests(self) -> bool:
        return bool(self._waiting or self._running)

    def num_waiting_requests(self) -> int:
        return len(self._waiting)

    def num_running_requests(self) -> int:
        return len(self._running)

    def get_request_state(self, request_id: str) -> DiffusionRequestState | None:
        return self._request_states.get(request_id)

    def pop_request_state(self, request_id: str) -> DiffusionRequestState | None:
        self._pop_extra_request_state(request_id)
        return self._request_states.pop(request_id, None)

    def preempt_request(self, request_id: str) -> bool:
        if request_id not in self._request_states:
            return False
        if request_id in self._running:
            self._running.remove(request_id)
            if not self._running:
                self._running_sampling_params_key = None
            state = self._request_states[request_id]
            state.status = DiffusionRequestStatus.PREEMPTED
            self._enqueue_waiting(state, front=True, count_arrival=False)
            return True
        return False

    def finish_requests(self, request_ids: str | list[str], status: DiffusionRequestStatus) -> None:
        assert DiffusionRequestStatus.is_finished(status)
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        self._finish_requests({request_id: status for request_id in request_ids})

    def close(self) -> None:
        self._request_states.clear()
        self._waiting.clear()
        self._waiting_by_key.clear()
        self._running.clear()
        self._running_sampling_params_key = None
        self._finished_req_ids.clear()
        self._key_arrivals.clear()
        self._arrival_window.clear()
        self._key_deficits.clear()
        self._drr_started = False
        self._drr_cursor_index = 0
        self._reset_scheduler_state()

    def _finish_requests(
        self,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        if not statuses:
            return set()

        finished_req_ids: set[str] = set()
        running_to_remove: set[str] = set()
        waiting_to_remove: set[str] = set()

        for request_id, status in statuses.items():
            assert DiffusionRequestStatus.is_finished(status)
            state = self._request_states.get(request_id)
            if state is None or state.is_finished():
                continue

            finished_req_ids.add(request_id)
            if request_id in self._running:
                running_to_remove.add(request_id)
            if request_id in self._waiting:
                waiting_to_remove.add(request_id)

        if running_to_remove:
            self._running = [request_id for request_id in self._running if request_id not in running_to_remove]
            if not self._running:
                self._running_sampling_params_key = None
        for request_id in waiting_to_remove:
            self._remove_waiting_id(request_id)

        for request_id in finished_req_ids:
            state = self._request_states[request_id]
            status = statuses[request_id]
            state.status = status
            if status == DiffusionRequestStatus.FINISHED_ERROR:
                state.error = None if errors is None else errors.get(request_id)
            else:
                state.error = None

        self._finished_req_ids |= finished_req_ids
        return finished_req_ids

    def _finalize_update_from_output(
        self,
        sched_output: DiffusionSchedulerOutput,
        statuses: dict[str, DiffusionRequestStatus],
        errors: dict[str, str | None] | None = None,
    ) -> set[str]:
        # A scheduled request may be aborted after schedule() but before
        # update_from_output() processes the runner output. It is already
        # marked finished at that point, but we still need to surface its id
        # in this update so the engine can observe the terminal state.
        finished_req_ids = {
            request_id for request_id in sched_output.scheduled_request_ids if request_id in self._finished_req_ids
        }
        finished_req_ids |= self._finish_requests(statuses, errors)
        return finished_req_ids

    def _reset_scheduler_state(self) -> None:
        """Reset subclass-owned state during initialize()/close()."""

    def _pop_extra_request_state(self, request_id: str) -> None:
        """Remove subclass-owned per-request state before popping request state."""

    def _make_request_state(self, request_id: str, request: OmniDiffusionRequest) -> DiffusionRequestState:
        return DiffusionRequestState(
            request_id=request_id,
            req=request,
            sampling_params_key=self._build_sampling_params_key(request),
        )

    def _resolve_batching_config(self, od_config: OmniDiffusionConfig) -> DiffusionBatchingConfig:
        cfg: Any = getattr(od_config, "diffusion_batching_config", None)
        if isinstance(cfg, DiffusionBatchingConfig):
            return cfg
        if isinstance(cfg, dict):
            return DiffusionBatchingConfig.from_dict(cfg)
        return DiffusionBatchingConfig()

    def _effective_max_running_reqs(self, candidate_state: DiffusionRequestState) -> int:
        """Return the request-count cap for the current homogeneous batch."""
        return self.batching_policy.get_max_running_reqs(self.max_num_running_reqs, candidate_state)

    def _enqueue_waiting(
        self,
        state: DiffusionRequestState,
        *,
        front: bool = False,
        count_arrival: bool = True,
    ) -> None:
        request_id = state.request_id
        if front:
            self._waiting.appendleft(request_id)
            self._waiting_by_key.setdefault(state.sampling_params_key, deque()).appendleft(request_id)
        else:
            self._waiting.append(request_id)
            self._waiting_by_key.setdefault(state.sampling_params_key, deque()).append(request_id)

        if count_arrival:
            self._record_key_arrival(state.sampling_params_key)
        self._key_deficits.setdefault(state.sampling_params_key, 0.0)

    def _record_key_arrival(self, key: SamplingParamsKey | None) -> None:
        self._arrival_window.append(key)
        self._key_arrivals[key] = self._key_arrivals.get(key, 0) + 1

        window_size = max(1, int(getattr(self.batching_config, "drr_arrival_window_size", 1024)))
        while len(self._arrival_window) > window_size:
            expired_key = self._arrival_window.popleft()
            new_count = self._key_arrivals.get(expired_key, 0) - 1
            if new_count > 0:
                self._key_arrivals[expired_key] = new_count
            else:
                self._key_arrivals.pop(expired_key, None)

    def _peek_next_waiting_state(self) -> DiffusionRequestState | None:
        if self._running:
            return self._peek_waiting_state_for_key(self._current_sampling_params_key())

        key = self._select_next_waiting_key()
        if key is None and key not in self._waiting_by_key:
            return None
        return self._peek_waiting_state_for_key(key)

    def _peek_waiting_state_for_key(self, key: SamplingParamsKey | None) -> DiffusionRequestState | None:
        queue = self._waiting_by_key.get(key)
        while queue:
            state = self._request_states.get(queue[0])
            if state is not None and not state.is_finished():
                return state
            queue.popleft()
        if key in self._waiting_by_key:
            self._waiting_by_key.pop(key, None)
        return None

    def _pop_waiting_state(self, state: DiffusionRequestState) -> None:
        self._remove_waiting_id(state.request_id)

    def _remove_waiting_id(self, request_id: str) -> None:
        self._waiting = deque(req_id for req_id in self._waiting if req_id != request_id)
        state = self._request_states.get(request_id)
        if state is None:
            return
        key = state.sampling_params_key
        queue = self._waiting_by_key.get(key)
        if queue is None:
            return
        self._waiting_by_key[key] = deque(req_id for req_id in queue if req_id != request_id)
        if not self._waiting_by_key[key]:
            self._waiting_by_key.pop(key, None)

    def _select_next_waiting_key(self) -> SamplingParamsKey | None:
        active_keys = []
        for key in list(self._waiting_by_key):
            queue = self._waiting_by_key.get(key)
            if queue and self._peek_waiting_state_for_key(key) is not None:
                active_keys.append(key)
        if not active_keys:
            return None

        active_keys.sort(key=lambda key: (self._key_cost(key), repr(key)))
        if not self._drr_started:
            self._drr_started = True
            selected = active_keys[0]
            self._drr_cursor_index = 1 % len(active_keys)
            return selected

        start = self._drr_cursor_index % len(active_keys)
        min_cost = self._key_cost(active_keys[0])
        active_arrivals = sum(self._key_arrivals.get(key, 0) for key in active_keys)
        for offset in range(len(active_keys)):
            index = (start + offset) % len(active_keys)
            key = active_keys[index]
            self._key_deficits[key] = self._key_deficits.get(key, 0.0) + self._key_quantum(
                key,
                min_cost,
                active_arrivals,
                len(active_keys),
            )
            if self._key_deficits[key] >= 1.0:
                self._key_deficits[key] -= 1.0
                self._drr_cursor_index = (index + 1) % len(active_keys)
                return key

        # All non-empty queues were visited but did not have enough budget.
        self._drr_cursor_index = start
        return None

    def _key_quantum(
        self,
        key: SamplingParamsKey | None,
        min_cost: int,
        active_arrivals: int,
        active_queue_count: int,
    ) -> float:
        # prob_i is estimated from a sliding arrival window. If all active queues
        # have aged out of the window, fall back to uniform probability to avoid
        # starving already-waiting requests.
        if active_arrivals <= 0:
            prob = 1.0 / max(1, active_queue_count)
        else:
            prob = self._key_arrivals.get(key, 0) / active_arrivals
        return prob * max(1, min_cost) / self._key_cost(key)

    def _key_cost(self, key: SamplingParamsKey | None) -> int:
        if key is None:
            return self.batching_config.spatial_compute_units(None, None, 1)
        units = self.batching_config.spatial_compute_units(key.height, key.width, key.num_frames)
        if key.do_classifier_free_guidance or (key.true_cfg_scale is not None and key.true_cfg_scale > 1):
            units = int(units * float(self.batching_config.cfg_compute_multiplier))
        return max(1, units)

    def _can_schedule_waiting(self, state: DiffusionRequestState) -> bool:
        if not self._running:
            return True

        current_key = self._current_sampling_params_key()
        return current_key is not None and current_key == state.sampling_params_key

    def _current_sampling_params_key(self) -> SamplingParamsKey | RequestBatchSamplingParamsKey | None:
        if self._running_sampling_params_key is not None or not self._running:
            return self._running_sampling_params_key
        state = self._request_states.get(self._running[0])
        self._running_sampling_params_key = None if state is None else state.sampling_params_key
        return self._running_sampling_params_key

    def _build_sampling_params_key(
        self, request: OmniDiffusionRequest
    ) -> SamplingParamsKey | RequestBatchSamplingParamsKey | None:
        return get_sampling_params_key(request)
