
def quantum_walk_generator(model, **kwargs):

    # Filter each key type.
    function = []
    iterable = []

    for key in kwargs:
        if callable(kwargs[key]):
            function.append(key)
        else:
            iterable.append(key)
            kwargs[key] = iter(kwargs[key])

    while (True):
        try:
            qw_kwargs = {key : next(kwargs[key])
                         for key in iterable}
        except StopIteration:
            return

        func_args = qw_kwargs.pop('func_args')
        for key in function:
            qw_kwargs[key] = kwargs[key](func_args)

        yield model(**qw_kwargs)
