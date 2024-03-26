print('graph {')

dim = 3

def out_of_bounds(axis):
    return axis < 0 or axis >= dim 

for y in range(-1, dim + 1):
    for x in range(-1, dim + 1):
        out_x = out_of_bounds(x)
        out_y = out_of_bounds(y)
        if out_x and out_y:
            continue

        node_str = ('\t"' + str((x, y)) + '" ['
                    + 'pos="' + str(1.75*x) + ',' + str(1.75*y) + '!" '
                    + 'width=0.75 height=0.75 fixedsize=True')
        if out_x or out_y:
            node_str += (' style="dashed" label="' +
                         str((x % dim, y % dim)) + '"')
        node_str += ']'

        print(node_str)

print()

for y in range(dim):
    for x in range(dim):
        tail = str((x, y))
        for d in [0, 2]:
            y_axis = d // 2
            shift = 1 if d % 2 == 0 else -1
            head = (str((x + shift, y)) if not y_axis else
                    str((x, y + shift)))

            arc_str = '\t "' + tail + '" -- "' + head + '";'

            print(arc_str)

        if x == 0:
            head = str((-1, y))
            arc_str = '\t "' + tail + '" -- "' + head + '";'
            print(arc_str)
        if y == 0:
            head = str((x, -1))
            arc_str = '\t "' + tail + '" -- "' + head + '";'
            print(arc_str)

print('}')
