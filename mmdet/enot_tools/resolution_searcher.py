from typing import Tuple

from enot.experimental.resolution_search.fixed_latency_search import ResolutionSearcherWithFixedLatencyIterator
from enot.experimental.resolution_search.resolution_strategy import ResolutionStrategy
from enot.latency import initialize_latency
from enot.latency import max_latency as max_search_space_latency
from enot.latency import min_latency as min_search_space_latency
from enot.latency import reset_latency


class ResolutionSearcher(ResolutionSearcherWithFixedLatencyIterator):

    def _update_latency(
            self,
            resolution_strategy: ResolutionStrategy,
    ) -> Tuple[int, int]:
        """Recomputes latency container with the provided resolution strategy and applies to the search space."""

        reset_latency(self._search_space)
        dataloader = resolution_strategy(self._dataloader)
        test_inputs = next(dataloader)

        model_args, model_kwargs = self._sample_to_model_inputs(test_inputs)

        # Latency calculators need the forward inputs  be torch.Tensors,
        # so we cahnge forward to it's dummy version here for latency initialization on new resolution.
        self._search_space.original_model.real_forward = self._search_space.original_model.forward
        self._search_space.original_model.forward = self._search_space.original_model.forward_dummy

        latency_container = initialize_latency(self._latency_type, self._search_space, model_args, model_kwargs)

        self._search_space.original_model.forward = self._search_space.original_model.real_forward
        min_latency = (min_search_space_latency(latency_container))
        max_latency = (max_search_space_latency(latency_container))
        return min_latency, max_latency
