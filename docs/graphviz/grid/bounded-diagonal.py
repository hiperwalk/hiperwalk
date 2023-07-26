print('digraph {')

dim = 3

def out_of_bounds(axis):
    return axis < 0 or axis >= dim 

for y in range(dim):
    for x in range(dim):
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

arc_count = 0
for y in range(dim):
    for x in range(dim):
        tail = str((x, y))
        for d in range(4):
            x_shift = 1 if d // 2 == 0 else -1
            y_shift = 1 if d % 2 == 0 else -1
            head = (x + x_shift, y + y_shift)

            if out_of_bounds(head[0]) or out_of_bounds(head[1]):
                continue

            arc_str = '\t "' + str(tail) + '" -> "' + str(head) + '"'
            arc_str += '[headlabel=' + str(arc_count)
            arc_str += ' labeldistance=2'
            arc_str += ' labelangle=-50'
            arc_str += '];'

            print(arc_str)
            arc_count += 1

print('}')
