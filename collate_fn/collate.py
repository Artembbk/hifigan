import logging
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {
        'audio': [],
        'spectrogram': [],
        'spectrogram_length': [],
        'audio_path': []
    }

    dataset_items = sorted(dataset_items, key=lambda x: -x['spectrogram'].shape[2])

    
    for item in dataset_items: 
        result_batch['audio'].append(item['audio'].squeeze(0))
        result_batch['spectrogram'].append(item['spectrogram'].squeeze(0).T)
        result_batch['audio_path'].append(item['audio_path'])
        result_batch['spectrogram_length'].append(result_batch['spectrogram'][-1].shape[0])

    
    result_batch['audio'] = pad_sequence(result_batch['audio'], True, 0)
    result_batch['spectrogram'] = pad_sequence(result_batch['spectrogram'], True, -11.5129251).transpose(1, 2)
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'])
    
    return result_batch