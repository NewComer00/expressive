import json
import os
from dataclasses import dataclass, asdict, fields, field
from typing import Any, Dict


def persist_to_file(filename: str):
    """
    A decorator to automatically save and load dataclass fields to/from a JSON file.
    Adds a `restore` method to reset fields to their default values.
    """
    def decorator(cls):
        def save(self):
            """
            Save the dataclass instance to a JSON file.
            """
            with open(filename, "w") as file:
                json.dump(asdict(self), file, indent=4)

        def load(self):
            """
            Load the dataclass instance from a JSON file.
            """
            if os.path.exists(filename):
                with open(filename, "r") as file:
                    data = json.load(file)
                    for field in fields(self):
                        setattr(self, field.name, data.get(
                            field.name, getattr(self, field.name)))

        def restore(self):
            """
            Restore the dataclass instance to its default values.
            """
            defaults = {f.name: f.default for f in fields(
                self) if f.default != field()}  # Get default values
            for key, value in defaults.items():
                setattr(self, key, value)
            self.save()  # Save the restored defaults

        # Attach save, load, and restore methods to the class
        cls.save = save
        cls.load = load
        cls.restore = restore
        return cls
    return decorator


if __name__ == '__main__':
    from dataclasses import dataclass, field
    from persist import persist_to_file

    @persist_to_file("user_data.json")
    @dataclass
    class UserData:
        name: str = field(default="Guest")  # Default value
        email: str = field(default="guest@example.com")  # Default value
        age: str = field(default="18")  # Default value

    # Create an instance of UserData
    user = UserData()

    # Load saved data (if any)
    user.load()

    # Print current data
    print(f"Loaded Data: {user}")

    # Update fields
    user.name = "John Doe"
    user.email = "john@example.com"
    user.age = "30"

    # Save the updated data
    user.save()
    print("Data saved!")

    # Print updated data
    print(f"Updated Data: {user}")

    # Restore to default values
    user.restore()
    print("Data restored to defaults!")
    print(f"Restored Data: {user}")
