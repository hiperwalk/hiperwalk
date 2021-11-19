# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 05:21:53 2014

@author: aaron
"""
import os
import config as cfg



def template_DTQW1D(inputNAME,outputNAME,extension):
    plt_FILE_NAME="HIPERWALK_TEMP_PLOT.plt"
    output = open(plt_FILE_NAME,'w')    

    output.write("reset\n")
    output.write("set grid\n")
    output.write("set xrange[%d:%d]\n"%(cfg.RANGEX[0],cfg.RANGEX[1]))
    output.write("set xlabel \"Position\"\n")
    output.write("set ylabel \"Probability\"\n")

    if cfg.PLOTTING_ZEROS==0:
        output.write("set datafile missing \"0.0000000000000000\"\n")




    if cfg.GRAPHTYPE=="CYCLE":    
        output.write("set title \"Quantum Walk on Cycle\"\n")
    elif cfg.GRAPHTYPE=="LINE":
        output.write("set title \"Quantum Walk on a Line\"\n")

    if extension=="PNG":
        output.write("set terminal png size 1000,500 enhanced font \"Helvetica,20\"\n")
    elif extension=="EPS":    
        output.write("set terminal postscript enhanced eps\n")

    output.write("set output \"%s\"\n"%(outputNAME))
    
    os.system( "paste HIPERWALK_TEMP_RANGE_1D.dat %s > HIPERWALK_TEMP_PDF.dat"%(inputNAME) )

    output.write("plot \"%s\" w l lw 2 lc \"black\" t \"\"\n"%("HIPERWALK_TEMP_PDF.dat"))

    output.close()

    os.system("gnuplot HIPERWALK_TEMP_PLOT.plt 2> /dev/null > /dev/null")


def template_DTQW2D(inputNAME,outputNAME,extension):

    output = open("HIPERWALK_TEMP_PLOT.plt",'w')    

    output.write("reset\n")
    output.write("set border 4095\n")
    
#    output.write("set datafile missing \"0.0000000000000000\"\n")    
    
    output.write("set xrange[%d:%d]\n"%(cfg.RANGEX[0],cfg.RANGEX[1]))
    output.write("set xlabel \"X\"\n")

    output.write("set yrange[%d:%d]\n"%(cfg.RANGEY[0],cfg.RANGEY[1]))
    output.write("set ylabel \"Y\"\n")    

    output.write("set zlabel \"Probability\" rotate by 90 offset -2\n")  

    
    output.write("set xtics offset -1,0\n")
    output.write("set ytics offset 0,-1\n")
    
    output.write("set ticslevel 0\n")

    output.write("set size 1,1\n")
    
    output.write("set view 60,60\n")
    
    output.write("set style data lines\n")
    
    output.write("set dgrid3d %d,%d,16\n"%(cfg.RANGEX[1]-cfg.RANGEX[0]+1,cfg.RANGEY[1]-cfg.RANGEY[0]+1))

    output.write("set isosamples 10000,10000; set samples 100,100;\n")

    if extension=="PNG":
        output.write("set terminal png size 1000,500 enhanced font \"Helvetica,20\"\n")
    elif extension=="EPS":    
        output.write("set terminal postscript color eps\n")
        
    output.write("set output \"%s\"\n"%(outputNAME))

    os.system( "paste HIPERWALK_TEMP_RANGE_2D.dat %s > HIPERWALK_TEMP_PDF"%(inputNAME) )


    output.write("splot \"HIPERWALK_TEMP_PDF\" using 1:2:3 linecolor rgb \"#336699\"  title \"\" with lines\n")    
    output.close()

    os.system("gnuplot HIPERWALK_TEMP_PLOT.plt 2> /dev/null > /dev/null")

def template_STAGGERED1D(inputNAME,outputNAME,extension):
    plt_FILE_NAME="HIPERWALK_TEMP_PLOT.plt"
    output = open(plt_FILE_NAME,'w')    

    output.write("reset\n")
    output.write("set grid\n")
    output.write("set xrange[%d:%d]\n"%(cfg.RANGEX[0],cfg.RANGEX[1]))
    output.write("set xlabel \"Position\"\n")
    output.write("set ylabel \"Probability\"\n")

    if cfg.PLOTTING_ZEROS==0:
        output.write("set datafile missing \"0.0000000000000000\"\n")

    output.write("set title \"Coinless Quantum Walk on Cycle\"\n")


    if extension=="PNG":
        output.write("set terminal png size 1000,500 enhanced font \"Helvetica,20\"\n")
    elif extension=="EPS":    
        output.write("set terminal postscript enhanced eps\n")

    output.write("set output \"%s\"\n"%(outputNAME))
    
    os.system( "paste HIPERWALK_TEMP_RANGE_1D.dat %s > HIPERWALK_TEMP_PDF.dat"%(inputNAME) )

    output.write("plot \"%s\" w l lw 2 lc \"black\" t \"\"\n"%("HIPERWALK_TEMP_PDF.dat"))
    
    output.close()

    os.system("gnuplot HIPERWALK_TEMP_PLOT.plt 2> /dev/null > /dev/null")

#    output.write("set ztics\n")

#    output.write("set mxtics 5\n")
#    output.write("set mytics 5\n")
#    output.write("set hidden3d\n")

#    output.write("set pm3d\n")    
#    output.write("set palette defined (0 \"blue\", 1 \"yellow\", 2 \"red\")\n")
#    output.write("set palette rgbformulae 33,13,10\n")
#    output.write("set palette model RGB\n")    
    
#    output.write("splot \"tmpfile\" using 1:2:3 title \"\" with pm3d\n")
#    output.write("splot \"tmpfile\" using 1:2:3 linecolor palette  title \"\" with lines\n")


 
    
    
def plotAnimation1D():
    print("[HIPERWALK] Creating animation, please wait.")



    for i in range( 0, cfg.STEPS + 1):
        if cfg.ALLSTATES:
            if i % cfg.SAVE_STATES_MULTIPLE_OF_N == 0:
                template_DTQW1D("NEBLINA_TEMP_PROB%05d.dat"%(i),"HIPERWALK_TEMP_PROB%05d.png"%(i),"PNG")
        else:
            template_DTQW1D("NEBLINA_TEMP_PROB%05d.dat"%(i),"HIPERWALK_TEMP_PROB%05d.png"%(i),"PNG")



            
    print( "[HIPERWALK] Generating animation: ................ evolution.gif")
    os.system("convert -delay %d -loop 0 \'HIPERWALK_TEMP_PROB*.png\' HIPERWALK_TEMP.gif"%cfg.DELAY)
    os.system("convert -layers Optimize HIPERWALK_TEMP.gif evolution.gif")  
    
    
def plotAnimation2D():
    print("[HIPERWALK] Creating animation, please wait.")

    for i in range( 0, cfg.STEPS + 1):
        if cfg.ALLSTATES:
            if i % cfg.SAVE_STATES_MULTIPLE_OF_N == 0:
                template_DTQW2D("NEBLINA_TEMP_PROB%05d.dat"%(i),"HIPERWALK_TEMP_PROB%05d.png"%(i),"PNG")
        else:
            template_DTQW2D("NEBLINA_TEMP_PROB%05d.dat"%(i),"HIPERWALK_TEMP_PROB%05d.png"%(i),"PNG")
 
    print( "[HIPERWALK] Generating animation: .................. evolution.gif")
    os.system("convert -delay %d -loop 0 \'HIPERWALK_TEMP_PROB*.png\' HIPERWALK_TEMP_TMP.gif"%cfg.DELAY)
    os.system("convert -layers Optimize HIPERWALK_TEMP_TMP.gif evolution.gif")  
        


###
### Statistical graphics
###

### 2D
def plotStatistics2d():

    print("[HIPERWALK] Generating statistical graphics with gnuplot...")

    generatingStatisticsPlotFile2D("HIPERWALK_TEMP_STANDARD_DEVIATION")
    os.system( "gnuplot HIPERWALK_TEMP_STANDARD_DEVIATION.plt 2> /dev/null > /dev/null")
    print("[HIPERWALK] Standard deviation: .................... standard_deviation.eps")    

    generatingStatisticsPlotFile2D("HIPERWALK_TEMP_MEAN_X")
    os.system( "gnuplot HIPERWALK_TEMP_MEAN_X.plt 2> /dev/null > /dev/null")
    print("[HIPERWALK] Mean(x): ............................... meanX.eps")

    generatingStatisticsPlotFile2D("HIPERWALK_TEMP_MEAN_Y")
    os.system( "gnuplot HIPERWALK_TEMP_MEAN_Y.plt 2> /dev/null > /dev/null")
    print("[HIPERWALK] Mean(y): ............................... meanY.eps")    


    if cfg.VARIANCE:
        generatingStatisticsPlotFile2D("HIPERWALK_TEMP_VARIANCE_X")
        os.system( "gnuplot HIPERWALK_TEMP_VARIANCE_X.plt 2> /dev/null > /dev/null")
        print("[HIPERWALK] Variance(x): ........................... varianceX.eps")    
        
        
        generatingStatisticsPlotFile2D("HIPERWALK_TEMP_VARIANCE_Y")
        os.system( "gnuplot HIPERWALK_TEMP_VARIANCE_Y.plt 2> /dev/null > /dev/null")
        print("[HIPERWALK] Variance(y): ........................... varianceY.eps")    
    
    
def generatingStatisticsPlotFile2D(statisticData):
    output = open("%s.plt"%(statisticData),'w')    
    output.write("reset\n")

    output.write("set fit logfile \'/dev/null\'\n")
    output.write("set term postscript eps enhanced\n")
    output.write("set termoption font \"Helvetica,22\"\n")

    output.write("set xlabel \"Time\"\n")
    output.write("f(x)=a*x+b\n")
    output.write("g(x)= a*x**2 + b*x +c\n")
    
    if( statisticData == "HIPERWALK_TEMP_MEAN_X" ):
        output.write("set title \"Quantum Walk Mean X\"\n")
        output.write("set ylabel \"Mean\"\n")
        output.write("fit f(x) \"./statistics.dat\" u 1:2 via a, b\n")
        output.write("set output \"meanX.eps\"\n")
        output.write("plot \"./statistics.dat\" u 1:2 t \"\", f(x) t sprintf(\"f(x)=ax+b, a=%.8f b=%.8f\", a, b )\n")

        
    elif( statisticData == "HIPERWALK_TEMP_MEAN_Y" ):
        output.write("set title \"Quantum Walk Mean Y\"\n")
        output.write("set ylabel \"Mean\"\n")
        output.write("fit f(x) \"./statistics.dat\" u 1:3 via a, b\n")
        output.write("set output \"meanY.eps\"\n")    
        output.write("plot \"./statistics.dat\" u 1:3 t \"\", f(x) t sprintf(\"f(x)=ax+b, a=%.8f b=%.8f\", a, b )\n")

    elif( statisticData == "HIPERWALK_TEMP_VARIANCE_X" ):
        output.write("set title \"Quantum Walk Variance X\"\n")
        output.write("set ylabel \"Variance\"\n")
        output.write("fit g(x) \"./statistics.dat\" u 1:4 via a, b,c\n")
        output.write("set output \"varianceX.eps\"\n")    
        output.write("plot \"./statistics.dat\" u 1:4 t \"\", g(x) t sprintf(\"a=%.5f b=%.f c=%.5f\", a, b,c)\n")

#        output.write("set title \"Quantum Walk Moment X\"\n")
#        output.write("set ylabel \"Moment\"\n")
#        output.write("fit g(x) \"./statistics.dat\" u 1:7 via a, b,c\n")
#        output.write("set output \"momentX.eps\"\n")    
#        output.write("plot \"./statistics.dat\" u 1:7 t \"\", g(x) t sprintf(\"a=%.5f b=%.f c=%.5f\", a, b,c)\n")



    elif( statisticData == "HIPERWALK_TEMP_VARIANCE_Y" ):
        output.write("set title \"Quantum Walk Variance Y\"\n")
        output.write("set ylabel \"Variance\"\n")
        output.write("fit g(x) \"./statistics.dat\" u 1:5 via a, b,c\n")
        output.write("set output \"varianceY.eps\"\n")    
        output.write("plot \"./statistics.dat\" u 1:5 t \"\", g(x) t sprintf(\"a=%.5f b=%.f c=%.5f\", a, b,c)\n")

#        output.write("set title \"Quantum Walk Moment Y\"\n")
#        output.write("set ylabel \"Moment\"\n")
#        output.write("fit g(x) \"./statistics.dat\" u 1:8 via a, b,c\n")
#        output.write("set output \"momentY.eps\"\n")    
#        output.write("plot \"./statistics.dat\" u 1:8 t \"\", g(x) t sprintf(\"a=%.5f b=%.f c=%.5f\", a, b,c)\n")
    
    elif(statisticData == "HIPERWALK_TEMP_STANDARD_DEVIATION"):
        output.write("set title \"Quantum Walk Standard Deviation\"\n")
        output.write("set ylabel \"Standard Deviation\"\n")
        output.write("fit f(x) \"./statistics.dat\" u 1:6 via a, b\n")
        output.write("set output \"standard_deviation.eps\"\n")
        output.write("plot \"./statistics.dat\" u 1:6 t \"\", f(x) t sprintf(\"f(x)=ax+b, a=%.3f b=%.3f\", a, b )\n")

    output.close()
    

### 1D

def plotStatistics1D():

    print("[HIPERWALK] Generating statistical graphics with gnuplot...")
    generatingStatisticsPlotFile1D("HIPERWALK_TEMP_STANDARD_DEVIATION")
    os.system( "gnuplot HIPERWALK_TEMP_STANDARD_DEVIATION.plt 2> /dev/null > /dev/null")
    print("[HIPERWALK] Standard deviation: ................. standard_deviation.eps")    

    generatingStatisticsPlotFile1D("HIPERWALK_TEMP_MEAN")
    os.system( "gnuplot HIPERWALK_TEMP_MEAN.plt 2> /dev/null > /dev/null")
    print("[HIPERWALK] Mean: ............................... mean.eps")
    
def generatingStatisticsPlotFile1D(statisticData):

    output = open("%s.plt"%statisticData,'w')    
    output.write("reset\n")

    output.write("set fit logfile \'/dev/null\'\n")
    output.write("set term postscript eps enhanced\n")
    output.write("set termoption font \"Helvetica,22\"\n")

    output.write("f(x)=a*x+b\n")

    if( statisticData == "HIPERWALK_TEMP_MEAN" ):
        output.write("fit f(x) \"./statistics.dat\" u 1:2 via a, b\n")
        if cfg.GRAPHTYPE=="CYCLE":    
            output.write("set title \"Quantum Walk Mean on Cycle\"\n")
        elif cfg.GRAPHTYPE=="LINE":
            output.write("set title \"Quantum Walk Mean on a Line\"\n")
        output.write("set output \"mean.eps\"\n")
        output.write("set ylabel \"Mean\"\n")
        output.write("plot \"./statistics.dat\" u 1:2 t \"\", f(x) t sprintf(\"f(x)=ax+b, a=%.3f b=%.3f\", a, b )\n")
    
    elif(statisticData == "HIPERWALK_TEMP_STANDARD_DEVIATION"):
        output.write("fit f(x) \"./statistics.dat\" u 1:4 via a, b\n")
        if cfg.GRAPHTYPE=="CYCLE":    
            output.write("set title \"Quantum Walk Standard Deviation on Cycle\"\n")
        elif cfg.GRAPHTYPE=="LINE":
            output.write("set title \"Quantum Walk Standard Deviation on a Line\"\n")
        output.write("set ylabel \"Standard Deviation\"\n")
        output.write("set output \"standard_deviation.eps\"\n")        
        output.write("plot \"./statistics.dat\" u 1:4 t \"\", f(x) t sprintf(\"f(x)=ax+b, a=%.3f b=%.3f\", a, b )\n")

    output.close()    