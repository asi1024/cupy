from cupy import _core


def create_math_ufunc(math_name, nargs, name, doc, support_complex=True):
    assert 1 <= nargs <= 2
    if nargs == 1:
        types = (
            ('e->e', f'out0 = {math_name}f(in0)'),
            ('f->f', f'out0 = {math_name}f(in0)'),
            'd->d',
        )
        if support_complex:
            types += ('F->F', 'D->D')
        return _core.create_ufunc(
            name, types, f'out0 = {math_name}(in0)', doc=doc)
    else:
        types = (
            ('ee->e', f'out0 = {math_name}f(in0, in1)'),
            ('ff->f', f'out0 = {math_name}f(in0, in1)'),
            'dd->d',
        )
        if support_complex:
            types += ('FF->F', 'DD->D')
        return _core.create_ufunc(
            name, types, f'out0 = {math_name}(in0, in1)', doc=doc)
