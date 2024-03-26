print('graph {')

dim = 3

def out_of_bounds(axis):
    return axis < 0 or axis >= dim 

for y in range(-1, dim + 1):
    for x in range(-1, dim + 1):
        out_x = out_of_bounds(x)
        out_y = out_of_bounds(y)

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
            x_shift = 1 if d // 2 == 0 else -1
            y_shift = 1 if d % 2 == 0 else -1
            head = str((x + x_shift, y + y_shift))

            arc_str = '\t "' + tail + '" -- "' + head + '";'
            print(arc_str)

        if x == 0 or y == 0:
            head = str((x -1, y - 1))
            arc_str = '\t "' + tail + '" -- "' + head + '";'
            print(arc_str)

        if y == 0 or x == dim - 1:
            head = str((x + 1, y - 1))
            arc_str = '\t "' + tail + '" -- "' + head + '";'
            print(arc_str)

print('}')
