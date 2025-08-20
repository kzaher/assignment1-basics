import pytest
import dataclasses
from typing import List, Dict, Any
from cs336_basics.extensions import replace_recursively


@dataclasses.dataclass
class InnerData:
    value: int
    name: str


@dataclasses.dataclass
class NestedData:
    inner: InnerData
    numbers: List[int]
    mapping: Dict[str, Any]


@dataclasses.dataclass
class ComplexData:
    nested: NestedData
    items: List[InnerData]
    metadata: Dict[str, InnerData]


class TestReplaceRecursively:
    """Test suite for replace_recursively function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.inner_data = InnerData(value=42, name="original")
        self.nested_data = NestedData(
            inner=self.inner_data,
            numbers=[1, 2, 3],
            mapping={"key1": "value1", "key2": self.inner_data},
        )
        self.complex_data = ComplexData(
            nested=self.nested_data,
            items=[
                InnerData(value=10, name="item1"),
                InnerData(value=20, name="item2"),
            ],
            metadata={
                "meta1": InnerData(value=100, name="meta_item1"),
                "meta2": InnerData(value=200, name="meta_item2"),
            },
        )

    def test_simple_attribute_replacement(self):
        """Test replacing a simple attribute in a dataclass."""
        new_inner = InnerData(value=999, name="new_name")
        result = replace_recursively(self.nested_data, lambda obj: obj.inner, new_inner)

        assert result.inner.value == 999
        assert result.inner.name == "new_name"
        # Verify original is unchanged
        assert self.nested_data.inner.value == 42
        assert self.nested_data.inner.name == "original"
        # Verify other fields are preserved
        assert result.numbers == [1, 2, 3]
        assert result.mapping == {"key1": "value1", "key2": self.inner_data}

    def test_nested_attribute_replacement(self):
        """Test replacing a nested attribute."""
        result = replace_recursively(self.nested_data, lambda obj: obj.inner.value, 777)

        assert result.inner.value == 777
        assert result.inner.name == "original"  # Should remain unchanged
        # Verify original is unchanged
        assert self.nested_data.inner.value == 42

    def test_list_element_replacement(self):
        """Test replacing an element in a list."""
        result = replace_recursively(self.nested_data, lambda obj: obj.numbers[1], 999)

        assert result.numbers == [1, 999, 3]
        # Verify original is unchanged
        assert self.nested_data.numbers == [1, 2, 3]

    def test_dict_value_replacement(self):
        """Test replacing a value in a dictionary."""
        result = replace_recursively(
            self.nested_data, lambda obj: obj.mapping["key1"], "new_value"
        )

        assert result.mapping["key1"] == "new_value"
        assert result.mapping["key2"] == self.inner_data
        # Verify original is unchanged
        assert self.nested_data.mapping["key1"] == "value1"

    def test_complex_nested_replacement(self):
        """Test replacing deeply nested values."""
        result = replace_recursively(
            self.complex_data, lambda obj: obj.nested.inner.value, 888
        )

        assert result.nested.inner.value == 888
        assert result.nested.inner.name == "original"
        # Verify original is unchanged
        assert self.complex_data.nested.inner.value == 42

    def test_list_of_dataclass_replacement(self):
        """Test replacing an item in a list of dataclasses."""
        new_item = InnerData(value=555, name="new_item")
        result = replace_recursively(
            self.complex_data, lambda obj: obj.items[0], new_item
        )

        assert result.items[0].value == 555
        assert result.items[0].name == "new_item"
        assert result.items[1].value == 20  # Should remain unchanged
        # Verify original is unchanged
        assert self.complex_data.items[0].value == 10

    def test_nested_dataclass_attribute_in_list(self):
        """Test replacing an attribute of a dataclass within a list."""
        result = replace_recursively(
            self.complex_data, lambda obj: obj.items[1].name, "modified_item"
        )

        assert result.items[1].name == "modified_item"
        assert result.items[1].value == 20  # Should remain unchanged
        assert result.items[0].name == "item1"  # Should remain unchanged
        # Verify original is unchanged
        assert self.complex_data.items[1].name == "item2"

    def test_dict_of_dataclass_replacement(self):
        """Test replacing a dataclass in a dictionary."""
        new_meta = InnerData(value=777, name="new_meta")
        result = replace_recursively(
            self.complex_data, lambda obj: obj.metadata["meta1"], new_meta
        )

        assert result.metadata["meta1"].value == 777
        assert result.metadata["meta1"].name == "new_meta"
        assert result.metadata["meta2"].value == 200  # Should remain unchanged
        # Verify original is unchanged
        assert self.complex_data.metadata["meta1"].value == 100

    def test_nested_dataclass_attribute_in_dict(self):
        """Test replacing an attribute of a dataclass within a dictionary."""
        result = replace_recursively(
            self.complex_data, lambda obj: obj.metadata["meta2"].value, 333
        )

        assert result.metadata["meta2"].value == 333
        assert result.metadata["meta2"].name == "meta_item2"  # Should remain unchanged
        assert result.metadata["meta1"].value == 100  # Should remain unchanged
        # Verify original is unchanged
        assert self.complex_data.metadata["meta2"].value == 200

    def test_immutability(self):
        """Test that the original object is not modified."""
        original_value = self.nested_data.inner.value
        original_name = self.nested_data.inner.name

        result = replace_recursively(self.nested_data, lambda obj: obj.inner.value, 999)

        # Original should be unchanged
        assert self.nested_data.inner.value == original_value
        assert self.nested_data.inner.name == original_name
        # Result should have new value
        assert result.inner.value == 999

    def test_deep_immutability(self):
        """Test that nested structures maintain immutability."""
        original_numbers = self.complex_data.nested.numbers.copy()
        original_inner_value = self.complex_data.nested.inner.value

        result = replace_recursively(
            self.complex_data, lambda obj: obj.nested.numbers[0], 999
        )

        # Original should be unchanged
        assert self.complex_data.nested.numbers == original_numbers
        assert self.complex_data.nested.inner.value == original_inner_value
        # Result should have new value
        assert result.nested.numbers[0] == 999

    def test_multiple_replacements(self):
        """Test performing multiple replacements in sequence."""
        # First replacement
        result1 = replace_recursively(
            self.complex_data, lambda obj: obj.nested.inner.value, 111
        )

        # Second replacement on the result
        result2 = replace_recursively(
            result1, lambda obj: obj.items[0].name, "updated_item"
        )

        assert result2.nested.inner.value == 111
        assert result2.items[0].name == "updated_item"
        # Original should still be unchanged
        assert self.complex_data.nested.inner.value == 42
        assert self.complex_data.items[0].name == "item1"

    def test_replacing_with_none(self):
        """Test replacing a value with None."""
        result = replace_recursively(
            self.nested_data, lambda obj: obj.mapping["key1"], None
        )

        assert result.mapping["key1"] is None
        assert result.mapping["key2"] == self.inner_data

    def test_replacing_entire_nested_structure(self):
        """Test replacing an entire nested dataclass structure."""
        new_nested = NestedData(
            inner=InnerData(value=999, name="new_nested"),
            numbers=[9, 8, 7],
            mapping={"new_key": "new_value"},
        )

        result = replace_recursively(
            self.complex_data, lambda obj: obj.nested, new_nested
        )

        assert result.nested.inner.value == 999
        assert result.nested.inner.name == "new_nested"
        assert result.nested.numbers == [9, 8, 7]
        assert result.nested.mapping == {"new_key": "new_value"}
        # Other fields should remain unchanged
        assert result.items == self.complex_data.items
        assert result.metadata == self.complex_data.metadata

    def test_lambda_function_variations(self):
        """Test different ways of writing the selector lambda."""
        # Test with explicit lambda
        result1 = replace_recursively(self.nested_data, lambda x: x.inner.value, 111)

        # Test with different variable name
        result2 = replace_recursively(
            self.nested_data, lambda data: data.inner.value, 111
        )

        # Both should produce the same result
        assert result1.inner.value == 111
        assert result2.inner.value == 111
        assert result1.inner.name == result2.inner.name

    def test_type_preservation(self):
        """Test that types are preserved after replacement."""
        result = replace_recursively(
            self.complex_data, lambda obj: obj.nested.inner.value, 555
        )

        # Check that the result is still of the correct type
        assert isinstance(result, ComplexData)
        assert isinstance(result.nested, NestedData)
        assert isinstance(result.nested.inner, InnerData)
        assert isinstance(result.nested.inner.value, int)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_level_dataclass(self):
        """Test with a simple single-level dataclass."""

        @dataclasses.dataclass
        class Simple:
            value: int

        obj = Simple(value=42)
        result = replace_recursively(obj, lambda x: x.value, 999)

        assert result.value == 999
        assert obj.value == 42  # Original unchanged

    def test_empty_collections(self):
        """Test with empty lists and dictionaries."""

        @dataclasses.dataclass
        class WithEmpty:
            empty_list: List[int]
            empty_dict: Dict[str, int]

        obj = WithEmpty(empty_list=[], empty_dict={})

        # Should work without errors even with empty collections
        result = replace_recursively(obj, lambda x: x.empty_list, [1, 2, 3])
        assert result.empty_list == [1, 2, 3]
        assert obj.empty_list == []  # Original unchanged


if __name__ == "__main__":
    pytest.main([__file__])
