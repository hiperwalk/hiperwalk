PYHBdir=~/projects/pyhiperblas-build
mkdir -p $PYHBdir
#python3.10 -m pip uninstall -y HiperblasExtension

#| Aspecto            | `-e/--editable`            | `--target`                        |
#| ------------------ | -------------------------- | --------------------------------- |
#| arquivos estão     | Link para fontes originais | Copiados para o diretório alvo    |
#| mudanças imediatas | ✅ Sim                     | ❌ Não                            |
#| Uso l              | Desenvolvimento            | Instalação isolada / distribuição |
#| Reinstalação       | ❌ Não                     | ✅ Sim, se fontes mudarem         |
#| Cria `.egg-link`   | ✅                         | ❌                                |


hwD=/prj/prjedlg/bidu/OneDrive/aLncc/passeiosQuantOut25/pyhiperblas
HOMEB=/mnt/c/Users/bidu/
hwD=${HOME}/OneDrive/aLncc/passeiosQuantOut25/pyhiperblas
hwD=/mnt/c/Users/bidu/OneDrive/aLncc/passeiosQuantOut25/pyhiperblas
hwD=${HOME}/OneDrive/aLncc/passeiosQuantOut25A/pyhiperblas
hwD=/mnt/c/Users/bidu/OneDrive/aLncc/passeiosQuantOut25A/pyhiperblas
hwD=${1:"."}
#cd $HOME/OneDrive/aLncc/passeiosQuantSet25/pyhiperblas
(
cd $hwD
#python3 -m pip install -e . --prefix $PYHBdir  --no-deps
comando="python3 -m pip install --user -e $hwD --break-system-packages --no-build-isolation --config-settings editable_mode=compat"  # linux puro
comando="python3 -m pip install --user -e $hwD --break-system-packages --no-deps --no-cache"  # wsl
echo $comando; eval $comando
)



#export PYTHONPATH=$PYTHONPATH:$PYHBdir
#python3 -m pip install    . --target $PYHBdir  --no-deps
#python3.10 -c "import hiperblas; print('import ok')"




