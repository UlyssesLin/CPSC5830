import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict

class LastMessageAggregator(nn.Module):
    def __init__(self, device):
        super().__init__(device)

    def aggregate(self, node_ids, messages):
        """Only keep the last message for each node"""
        unique_node_ids = np.unique(node_ids)

        to_update_node_ids = []
        unique_messages = []
        unique_timestamps = []

        for node_id in unique_node_ids:
            if len(messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_messages.append(messages[node_id][-1][0])
                unique_timestamps.append(messages[node_id][-1][1])

        if len(to_update_node_ids) > 0:
            unique_messages = torch.stack(unique_messages)
            unique_timestamps = torch.stack(unique_timestamps)

        return to_update_node_ids, unique_messages, unique_timestamps