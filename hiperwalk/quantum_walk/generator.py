
def quantum_walk_generator(model, **kwargs):
    """
    Create a generator of quantum walks.

    Parameters
    ----------
    model: :class:`QuantumWalk` class
        Quantum walk model for which multiple instances will be created.
        A :class:`QuantumWalk` class is expected, not an object.

    **kwargs:
        Arguments to instantiate a quantum walk.
        The arguments are passed to
        the constructor of the given quantum walk model.
        Each key must correspond to a valid constructor key.
        One additional key is also available: ``func_args``,
        explained in detail later.
        There are two types of valid keys:
        function, and iterable.
        The types of keys can be used simultaneously.

        function:
            A function that generates a constructor kwarg on demand.
            If a function is passed,
            the ``func_args`` key must also be specified.
            ``func_args`` must be any iterable such that
            its values are valid arguments for ``function``.
            It is recommended to use a generator as ``func_args``
            to optimize memory usage.

        iterable:
            A quantum walk instance is created for each iterable entry,
            even when multiple iterables are specified.
            For example, if ``iter1`` and ``iter2``
            are specified in two different kwargs,
            the values of ``iter1[i]`` and ``iter2[i]`` are used to
            create the ``i``-th quantum walk.
            It is recommended to use a generator as the iterable
            to optimize memory usage.

    Examples
    --------
    The following creates coined quantum walks on hypercubes with
    increasing dimension.

    .. testsetup::

        from sys import path
        path.append('../..')
        import hiperwalk as hpw

    >>> gen = hpw.quantum_walk_generator(hpw.Coined,
    ...             graph=hpw.Hypercube,
    ...             marked=[0, 1, 2, 3, 4],
    ...             func_args=range(5, 10))

    This creates five coined quantum walks on hypercubes.
    The ``func_args`` values are used to invoke any
    callable keyword, like the hypercube constructor.
    Hence, the first quantum walk occurs on a 5-dimensional hypercube,
    the second on a 6-dimensional hypercube, and so on.
    Since ``marked`` is an iterable,
    it changes simultaneously with the other iterables, like ``func_args``.
    Hence, vertex 0 is marked on the first quantum walk,
    vertex 1 is marked on the second quantum walk, and so on.

    >>> gen #doctest: +SKIP
    <generator object quantum_walk_generator at 0x7fda7e13af80>
    >>> qw_list = [qw for qw in gen]
    >>> [qw._graph.dimension() for qw in qw_list] == list(range(5, 10))
    True
    >>> [qw.get_marked()[0] for qw in qw_list] == [0, 1, 2, 3, 4]
    True

    Notes
    -----
    A fixed key type is not implemented to avoid ambiguity.
    For example, in a quantum walk on a grid, if ``marked=(1, 1)``,
    it is unclear whether
    vertex ``(1, 1)`` should be marked in all quantum walks or
    vertex ``1`` should be marked in two quantum walks.
    """

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
