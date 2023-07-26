print('digraph {')

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

arc_count = 0
for y in range(dim):
    for x in range(dim):
        tail = str((x, y))
        for d in range(4):
            y_axis = d // 2
            shift = 1 if d % 2 == 0 else -1
            head = (str((x + shift, y)) if not y_axis else
                    str((x, y + shift)))

            arc_str = '\t "' + tail + '" -> "' + head + '"'
            arc_str += '[headlabel=' + str(arc_count)
            arc_str += ' labeldistance=4'
            arc_str += ' labelangle=-20'
            arc_str += '];'

            print(arc_str)
            arc_count += 1

print('}')
