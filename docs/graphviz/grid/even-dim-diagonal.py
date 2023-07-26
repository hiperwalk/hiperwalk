print('digraph {')

dim = 4

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

        if (x % dim + y % dim) % 2:
            node_str += ' fontcolor="white" fillcolor="red" style="filled'
            node_str += ',dashed" penwidth=2' if out_x or out_y else '"'

        node_str += ']'

        print(node_str)

print()

arc_count = 0
for y in range(dim):
    for x in range(dim):
        tail = str((x, y))
        for d in range(4):
            x_shift = 1 if d // 2 == 0 else -1
            y_shift = 1 if d % 2 == 0 else -1
            head = str((x + x_shift, y + y_shift))

            arc_str = '\t "' + tail + '" -> "' + head + '"'
            arc_str += '['
            #arc_str += 'headlabel=' + str(arc_count)
            if (x % dim + y % dim) % 2:
                arc_str += ' color="red"'
                arc_str += ' style="dashed"'
            arc_str += ' labeldistance=2'
            arc_str += ' labelangle=-50'
            arc_str += ' penwidth=2'
            arc_str += '];'

            print(arc_str)
            arc_count += 1

print('}')
