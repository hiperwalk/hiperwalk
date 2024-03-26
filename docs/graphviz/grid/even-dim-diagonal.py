print('graph {')

dim = 4

def out_of_bounds(vertex):
    return (vertex[0] < 0 or vertex[0] >= dim
            or
            vertex[1] < 0 or vertex[1] >= dim)

for y in range(-1, dim + 1):
    for x in range(-1, dim + 1):

        node_str = ('\t"' + str((x, y)) + '" ['
                    + 'pos="' + str(1.75*x) + ',' + str(1.75*y) + '!" '
                    + 'width=0.75 height=0.75 fixedsize=True')

        out = out_of_bounds((x, y))
        if out:
            node_str += (' style="dashed" label="' +
                         str((x % dim, y % dim)) + '"')

        if (x % dim + y % dim) % 2:
            node_str += ' fontcolor="white" fillcolor="darkgray" style="filled'
            #node_str += ',dashed" penwidth=2' if out else '"'
            node_str += ',dashed"' if out else '"'

        node_str += ']'

        print(node_str)

print()

arc_count = 0
for y in range(-1, dim + 1):
    for x in range(-1, dim + 1):
        for d in [0, 2]:
            tail = (x, y)

            x_shift = 1 if d // 2 == 0 else -1
            y_shift = 1 if d % 2 == 0 else -1
            head = (x + x_shift, y + y_shift)

            if out_of_bounds(head) and out_of_bounds(tail):
                continue

            tail = str(tail)
            head = str(head)

            arc_str = '\t "' + tail + '" -- "' + head + '"'
            arc_str += '['
            #arc_str += 'headlabel=' + str(arc_count)
            if (x % dim + y % dim) % 2:
                arc_str += ' color="black"'
                arc_str += ' style="dashed"'
            # arc_str += ' penwidth=2'
            arc_str += '];'

            print(arc_str)
            arc_count += 1

print('}')
