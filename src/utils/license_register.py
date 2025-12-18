# Copyright 2025 ODesign Team and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
License registration utilities for ODesign project.
"""
import inspect
from functools import wraps
from typing import Callable, Optional


# Default license header following Apache 2.0 format
DEFAULT_LICENSE_HEADER = """Copyright 2025 ODesign Team and/or its affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""


# Preset license configurations
LICENSE_PRESETS = {
    'bytedance2024': {
        'copyright_year': 2024,
        'copyright_holder': 'ByteDance and/or its affiliates',
        'license_type': 'Apache License, Version 2.0',
        'include_full_text': True,
    },
    'odesign2025': {
        'copyright_year': 2025,
        'copyright_holder': 'ODesign Team and/or its affiliates',
        'license_type': 'Apache License, Version 2.0',
        'include_full_text': True,  
    },
}


def register_license(
    preset_or_year = None,
    copyright_holder: str = None,
    license_type: str = "Apache License, Version 2.0",
    include_full_text: bool = True,
) -> Callable:
    """
    Decorator to automatically register license information for functions and classes.
    
    This decorator adds license declaration before the docstring and stores license 
    metadata in the function/class object. For classes, it directly modifies the class 
    without wrapping. For functions, it creates a wrapper to preserve functionality.
    
    Args:
        preset_or_year: Either a preset name string ('bytedance2024', 'odesign2025') 
                       or a copyright year (int). If None, defaults to 2024.
        copyright_holder: Copyright holder, defaults to "ByteDance and/or its affiliates"
        license_type: License type, defaults to "Apache License, Version 2.0"
        include_full_text: Whether to include full Apache 2.0 license text
    
    Returns:
        Decorated function or class with license information
        
    Example:
        >>> # For functions
        >>> @register_license('bytedance2024')
        >>> def my_function():
        ...     '''My function description'''
        ...     pass
        
        >>> # For classes
        >>> @register_license('odesign2025')
        >>> class MyClass:
        ...     '''My class description'''
        ...     pass
    """
    # Handle preset configurations
    if isinstance(preset_or_year, str):
        if preset_or_year in LICENSE_PRESETS:
            preset = LICENSE_PRESETS[preset_or_year]
            copyright_year = preset['copyright_year']
            copyright_holder = preset['copyright_holder']
            license_type = preset['license_type']
            include_full_text = preset['include_full_text']
        else:
            raise ValueError(f"Unknown license preset: {preset_or_year}. Available presets: {list(LICENSE_PRESETS.keys())}")
    else:
        # Use year or default
        copyright_year = preset_or_year if preset_or_year is not None else 2024
        if copyright_holder is None:
            copyright_holder = "ByteDance and/or its affiliates"
    
    def decorator(func_or_class: Callable) -> Callable:
        # Build license text
        if include_full_text and license_type == "Apache License, Version 2.0":
            license_text = f"""Copyright {copyright_year} {copyright_holder}.

            Licensed under the Apache License, Version 2.0 (the "License");
            you may not use this file except in compliance with the License.
            You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

            Unless required by applicable law or agreed to in writing, software
            distributed under the License is distributed on an "AS IS" BASIS,
            WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
            See the License for the specific language governing permissions and
            limitations under the License."""
        else:
            license_text = f"Copyright {copyright_year} {copyright_holder}.\n"
            license_text += f'Licensed under the {license_type} (the "License");'
        
        # Check if decorating a class or a function
        is_class = inspect.isclass(func_or_class)
        
        if is_class:
            # For classes: directly add metadata without wrapping
            func_or_class.__license__ = license_text
            func_or_class.__copyright__ = f"Copyright {copyright_year} {copyright_holder}"
            func_or_class.__license_type__ = license_type
            
            # Prepend license to class docstring if exists
            if func_or_class.__doc__:
                original_doc = func_or_class.__doc__
                func_or_class.__doc__ = f"{license_text}\n\n{original_doc}"
            else:
                func_or_class.__doc__ = license_text
            
            return func_or_class
        else:
            # For functions/methods: add metadata and use wraps
            func_or_class.__license__ = license_text
            func_or_class.__copyright__ = f"Copyright {copyright_year} {copyright_holder}"
            func_or_class.__license_type__ = license_type
            
            # Prepend license to function docstring if exists
            if func_or_class.__doc__:
                original_doc = func_or_class.__doc__
                func_or_class.__doc__ = f"{license_text}\n\n{original_doc}"
            else:
                func_or_class.__doc__ = license_text
            
            @wraps(func_or_class)
            def wrapper(*args, **kwargs):
                return func_or_class(*args, **kwargs)
            
            # Preserve license information on wrapper
            wrapper.__license__ = func_or_class.__license__
            wrapper.__copyright__ = func_or_class.__copyright__
            wrapper.__license_type__ = func_or_class.__license_type__
            
            return wrapper
    
    return decorator


def add_license_to_function_docstring(
    func: Callable,
    copyright_year: Optional[int] = 2024,
    copyright_holder: str = "ByteDance and/or its affiliates",
    license_type: str = "Apache License, Version 2.0",
    include_full_text: bool = True,
) -> None:
    """
    Directly add license information to an existing function's docstring.
    
    Args:
        func: Function to add license to
        copyright_year: Copyright year
        copyright_holder: Copyright holder
        license_type: License type
        include_full_text: Whether to include full Apache 2.0 license text
    """
    if include_full_text and license_type == "Apache License, Version 2.0":
        license_text = f"""Copyright {copyright_year} {copyright_holder}.

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License."""
    else:
        license_text = f"Copyright {copyright_year} {copyright_holder}.\n"
        license_text += f'Licensed under the {license_type} (the "License");'
    
    if func.__doc__:
        # Check if license already exists
        if "Copyright" not in func.__doc__:
            original_doc = func.__doc__
            func.__doc__ = f"{license_text}\n\n{original_doc}"
    else:
        func.__doc__ = license_text
    
    # Add metadata
    func.__license__ = license_text
    func.__copyright__ = f"Copyright {copyright_year} {copyright_holder}"
    func.__license_type__ = license_type


def get_license_header(
    copyright_year: Optional[int] = 2024,
    copyright_holder: str = "ByteDance and/or its affiliates",
    license_type: str = "Apache License, Version 2.0",
    as_comment: bool = True,
    include_full_text: bool = True,
) -> str:
    """
    Get formatted license header text.
    
    Args:
        copyright_year: Copyright year
        copyright_holder: Copyright holder
        license_type: License type
        as_comment: Whether to return as Python comment format (# prefix)
        include_full_text: Whether to include full Apache 2.0 license text
    
    Returns:
        Formatted license header string
        
    Example:
        >>> header = get_license_header()
        >>> print(header)
        # Copyright 2024 ByteDance and/or its affiliates.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # ...
    """
    if include_full_text and license_type == "Apache License, Version 2.0":
        license_text = f"""Copyright {copyright_year} {copyright_holder}.

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License."""
    else:
        license_text = f"Copyright {copyright_year} {copyright_holder}.\n"
        license_text += f'Licensed under the {license_type} (the "License");'
    
    if as_comment:
        lines = license_text.split('\n')
        commented_lines = [f"# {line}" if line.strip() else "#" for line in lines]
        return '\n'.join(commented_lines)
    
    return license_text


def check_license(func: Callable) -> bool:
    """
    Check if a function has already registered license information.
    
    Args:
        func: Function to check
    
    Returns:
        True if function has license information, False otherwise
    """
    return hasattr(func, '__license__') or (
        func.__doc__ and "Copyright" in func.__doc__ and "Licensed" in func.__doc__
    )


# Register license for current module
__license__ = DEFAULT_LICENSE_HEADER
__copyright__ = "Copyright 2024 ByteDance and/or its affiliates"
