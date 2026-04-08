coins=("G" "F")
hpcs=("none" "cpu" "gpu")
touch output.txt
rm output.txt

for ((dim=4; dim<=19; dim++)); do
	for j in "${!coins[@]}"; do
		coin=${coins[$j]}
		for ((state=0; state<=1; state++)); do
			for k in "${!hpcs[@]}"; do
				hpc=${hpcs[$k]}
				if [ $state -eq 0 ]; then
					echo $dim, $coin, 'ket0', $hpc
					echo $dim, $coin, 'ket0', $hpc >> output.txt
				else
					echo $dim, $coin, 'unif', $hpc
					echo $dim, $coin, 'unif', $hpc >> output.txt
				fi
				for ((execution=1; execution<=30; execution++));do
					echo "Execution" $execution
					echo "Execution" $execution >> output.txt
					python3 hypercube.py $dim $coin $state $hpc >> output.txt
				done
				echo
			done
		done
	done
done
