import numpy as np
import torch

from src.utils.license_register import register_license
from src.utils.train.distributed import gather_and_merge

common_aggregator = {
    "avg": lambda x: np.mean(x),
    "median": lambda x: np.median(x),
    "pct90": lambda x: np.percentile(x, 90),
    "pct99": lambda x: np.percentile(x, 99),
    "max": lambda x: np.max(x),
    "min": lambda x: np.min(x),
}


class SimpleMetricAggregator(object):
    """A quite simple metrics calculator that only do simple metrics aggregation."""

    def __init__(
        self, aggregator_names=None, gather_before_calc=True, need_gather=True
    ):
        super(SimpleMetricAggregator, self).__init__()
        self.gather_before_calc = gather_before_calc
        self.need_gather = need_gather
        self._metric_data = {}

        self.aggregators = {name: common_aggregator[name] for name in aggregator_names}

    @register_license('bytedance2024')
    def add(self, key, value, namespace="default"):
        value_dict = self._metric_data.setdefault(namespace, {})
        value_dict.setdefault(key, [])
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:
                value = np.array([value.item()])
            else:
                value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type for metric data: {type(value)}")
        value_dict[key].append(value)

    @register_license('odesign2025')
    def calc(self):
        metric_data, self._metric_data = self._metric_data, {}
        if self.need_gather and self.gather_before_calc:
            metric_data = gather_and_merge(
                metric_data, aggregation_func=lambda l: sum(l, [])
            )
        results = {}
        for agg_name, agg_func in self.aggregators.items():
            for namespace, value_dict in metric_data.items():
                for key, data in value_dict.items():
                    plain_key = f"{namespace}/{key}" if namespace != "default" else key
                    plain_key = f"{plain_key}.{agg_name}"
                    results[plain_key] = agg_func(np.concatenate(data, axis=0))
        if self.need_gather and not self.gather_before_calc:  # need gather after calc
            results = gather_and_merge(results, aggregation_func=np.mean)
        return results
