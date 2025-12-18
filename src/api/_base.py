import torch
from typing import NewType, get_origin, get_args, Union
import attr

# Create distinct types that can be distinguished at runtime
LogitsType = NewType('LogitsType', torch.Tensor)  # Tensor of Float, [N, C]
EmbeddingType = NewType('EmbeddingType', torch.Tensor)  # Tensor of Float, [N, D]
PairwiseEmbeddingType = NewType('PairwiseEmbeddingType', torch.Tensor)  # Tensor of Float, [N, N, D]
MaskType = NewType('MaskType', torch.Tensor)  # Tensor of Boolean, [N, ...]
IndexType = NewType('IndexType', torch.Tensor)  # Tensor of Integer, [N]
OneHotType = NewType('OneHotType', torch.Tensor)  # Tensor of Integer, [N, ...]
FlagType = NewType('FlagType', torch.Tensor)  # Tensor of Integer, [N, ...] nn.Embedding
FeatureType = NewType('FeatureType', torch.Tensor)  # Tensor of Float, [N, D] nn.Linear

def to_mask_type(tensor: torch.Tensor) -> MaskType:
    """Convert tensor to boolean mask type"""
    return tensor.bool()

def to_index_type(tensor: torch.Tensor) -> IndexType:
    """Convert tensor to integer index type, preserving existing integer types"""
    if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.bool]:
        return tensor.long()
    return tensor  # Keep existing integer types (int32, int64)

def to_embedding_type(tensor: torch.Tensor) -> EmbeddingType:
    """Convert tensor to float embedding type, preserving bf16/fp16"""
    if tensor.dtype in [torch.int32, torch.int64, torch.long, torch.bool]:
        return tensor.float()
    return tensor

def to_onehot_type(tensor: torch.Tensor) -> OneHotType:
    """Convert tensor to integer one-hot type"""
    if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        class_num = tensor.shape[-1]
        return torch.nn.functional.one_hot(tensor.argmax(-1), class_num).long()
    elif tensor.dtype == torch.bool:
        return tensor.long()
    return tensor  # Keep existing integer types

def to_pairwise_embedding_type(tensor: torch.Tensor) -> PairwiseEmbeddingType:
    """Convert tensor to float pairwise embedding type, preserving bf16/fp16"""
    if tensor.dtype in [torch.int32, torch.int64, torch.long, torch.bool]:
        return tensor.float()
    return tensor

def to_logits_type(tensor: torch.Tensor) -> LogitsType:
    """Convert tensor to float logits type, preserving bf16/fp16"""
    if tensor.dtype in [torch.int32, torch.int64, torch.long, torch.bool]:
        return tensor.float()
    return tensor

def to_flag_type(tensor: torch.Tensor) -> FlagType:
    """Convert tensor to int flag type, preserving existing integer types"""
    if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        return tensor.long()
    elif tensor.dtype == torch.bool:
        return tensor.long()
    return tensor  # Keep existing integer types

def to_feature_type(tensor: torch.Tensor) -> FeatureType:
    """Convert tensor to float feature type, preserving bf16/fp16"""
    if tensor.dtype in [torch.int32, torch.int64, torch.long, torch.bool]:
        return tensor.float()
    return tensor



def auto_type_convert(cls):
    """Decorator to automatically add type conversion to dataclass"""
    
    # For attrs classes, we need to use __attrs_post_init__
    original_attrs_post_init = getattr(cls, '__attrs_post_init__', None)
    original_post_init = getattr(cls, '__post_init__', None)
    
    def convert_types(self):        
        # Get type hints
        type_hints = getattr(cls, '__annotations__', {})
        
        # Type converter mapping
        type_converter_map = {
            'MaskType': to_mask_type,
            'IndexType': to_index_type,
            'EmbeddingType': to_embedding_type,
            'OneHotType': to_onehot_type,
            'PairwiseEmbeddingType': to_pairwise_embedding_type,
            'LogitsType': to_logits_type,
            'FlagType': to_flag_type,
        }
        
        # Apply conversions based on type hints
        for field_name, field_type in type_hints.items():
            if hasattr(self, field_name):
                current_value = getattr(self, field_name)
                if current_value is not None:
                    # Get the type name
                    if get_origin(field_type) is Union: # Handle Union types like IndexType | None
                        field_type = get_args(field_type)[0]
                    type_name = getattr(field_type, '__name__', str(field_type))
                    
                    if type_name in type_converter_map:
                        converter = type_converter_map[type_name]
                        converted_value = converter(current_value)
                        setattr(self, field_name, converted_value)
                    else:
                        setattr(self, field_name, current_value)
    
    def __attrs_post_init__(self):
        convert_types(self)
        # Call original __attrs_post_init__ if it exists
        if original_attrs_post_init:
            original_attrs_post_init(self)
    
    def __post_init__(self):
        convert_types(self)
        # Call original __post_init__ if it exists
        if original_post_init:
            original_post_init(self)
    
    # Set both methods to handle different dataclass implementations
    cls.__attrs_post_init__ = __attrs_post_init__
    cls.__post_init__ = __post_init__
    
    return cls


class DictAccessMixin:
    """Mixin to add dict-like access to any class"""
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def keys(self):
        # For attrs classes, get field names from class definition
        if hasattr(attr, 'fields') and attr.has(self.__class__):
            return [field.name for field in attr.fields(self.__class__)]
        return vars(self).keys()
    
    def values(self):
        return [getattr(self, key) for key in self.keys()]
    
    def items(self):
        return [(key, getattr(self, key)) for key in self.keys()]
    
    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, other=None, **kwargs):
        """Update attributes from dict or keyword arguments
        
        Usage:
            obj.update({'attr1': value1, 'attr2': value2})
            obj.update(attr1=value1, attr2=value2)
        """
        if other is not None:
            for key, value in other.items():
                if hasattr(self, key) and value is not None:
                    setattr(self, key, value)
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict:
        """Convert this object to a dictionary.
        
        Args:
            deep (bool): If True, recursively convert nested DictAccessMixin or dataclasses to dicts.
        """
        return {key: getattr(self, key) for key in self.keys()}