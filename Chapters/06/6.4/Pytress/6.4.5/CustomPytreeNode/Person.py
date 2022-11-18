import jax


class Person:

    # The first parameter of constructor should be the name of custom class.
    def __init__(self, name: str, birthday: int, phone: str, address: str):

        self.name = name
        self.birthday = birthday
        self.phone = phone
        self.address = address


def flatten(container: Person):

    contents = [container.name, container.birthday, container.phone, container.address]
    auxillary = container.name

    # The order of return value is fixed, could not be changed
    return contents, auxillary

def unflatten(auxillary: str, contents: list) -> Person:

    # Here the auto unboxing of Python is used
    return Person(auxillary, contents)

def start():

    leaves = jax.tree_util.tree_leaves([

        Person(name = "Zhang Hui", birthday = 20020913, phone = "17249888005", address = "Beijing"),
        Person(name = "Li Zhong", birthday = 20010913, phone = "17249888005", address = "Beijing"),
        Person(name = "Zhang Qun", birthday = 20030913, phone = "17249888005", address = "Beijing"),

    ])


    print(leaves)

if __name__ == '__main__':

    start()
