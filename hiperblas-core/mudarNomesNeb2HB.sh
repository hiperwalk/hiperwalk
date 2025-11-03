for f in $(grep -R neblina * -l); do sed 's/neblina/hiperblas/g' $f -i ; done
ls */*neblina* > lA; cp lA lB
f=lB; sed -i 's/neblina/hiperblas/g' $f
paste lA lB | awk '{print "mv ", $0}' |bash

