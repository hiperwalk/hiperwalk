import pandas as pd
import re

def parse_file(filepath):
    rows = []

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    current_header = None

    while i < len(lines):
        line = lines[i]

        # Detect header line
        if re.match(r'^\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+$', line):
            parts = [x.strip() for x in line.split(',')]
            current_header = {
                'dim': parts[0],
                'coin': parts[1],
                'state': parts[2],
                'hpc': parts[3],
            }
            i += 1
            continue

        # Detect execution
        exec_match = re.match(r'Execution\s+(\d+)', line)
        if exec_match and current_header:
            execution = int(exec_match.group(1))

            metrics = {}
            i += 1

            # Read metrics until next Execution or header
            while i < len(lines):
                next_line = lines[i]

                if next_line.startswith("Execution") or ',' in next_line:
                    break

                # metric_match = re.match(r'(.+?):\s*([\d\.]+)', next_line)
                metric_match = re.match(r'(.+?):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
                                        next_line)
                if metric_match:
                    key = metric_match.group(1).strip().lower()
                    value = float(metric_match.group(2))
                    metrics[key] = value

                i += 1

            # Save row
            row = {
                **current_header,
                'execution': execution,
                'create graph': metrics.get('create graph'),
                'create qw': metrics.get('create qw'),
                'simulation': metrics.get('simulation'),
            }

            rows.append(row)
            continue

        i += 1

    df = pd.DataFrame(rows)
    return df


# Example usage
if __name__ == "__main__":
    df = parse_file("output.txt")
    # Optional: save to CSV
    df.to_csv("output.csv", index=False)
    df = df.drop(columns=['execution'])
    df = df.groupby(['dim', 'coin', 'state', 'hpc'], as_index=False).mean()

    df = df.pivot_table(
            index=['dim', 'coin', 'state'],
            columns='hpc',
            values=['create graph', 'create qw', 'simulation']
        )
    df['diff create graph'] = df[('create graph', '0')] - df[('create graph', '1')]
    df['diff create qw'] = df[('create qw', '0')] - df[('create qw', '1')]
    df['diff simulation'] = df[('simulation', '0')] - df[('simulation', '1')]

    df = df.reset_index()[['dim', 'coin', 'state', 'diff create graph',
                           'diff create qw', 'diff simulation']]
    df.columns = df.columns.droplevel('hpc')
    df['dim'] = df['dim'].astype(int)
    df = df.sort_values(by='dim', ascending=True)

    df['diff total'] = df['diff create graph'] + df['diff create qw'] + df['diff simulation']

    print(df.columns)

    df.to_csv('output_avg.csv', index=False)

    # df = df[df['dim'] >= 13] # for create qw
    # df = df[df['dim'] >= 10] # for simulation
    df = df[df['dim'] >= 11] # for diff total
    print(df.head())

    import matplotlib.pyplot as plt

    for (coin, state), group in df.groupby(['coin', 'state']):
        group = group.sort_values('dim')
        plt.plot(group['dim'], group['diff total'], marker='o',
                              label=f'{coin}, {state}')

        plt.yscale('log')
        plt.xlabel('dim')
        plt.ylabel('diff total (s)')
        plt.legend()

    plt.savefig('total')
