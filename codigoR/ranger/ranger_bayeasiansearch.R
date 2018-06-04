# Ranger 
# Montecarlo Estimation
# Búsqueda de parámetros óptimos con Bayesian Search, más inteligente que Grid Search
# Optimizo  Lift

# Limpio la memoria
rm( list=ls() )
gc()


# source( "~/cloud/cloud1/codigoR/ranger/ranger_bayesiansearch.R" )
library(ranger)
library(caret)
library(ROCR)
library("rBayesianOptimization" )


# Establezco la carpeta de trabajo
# setwd( "~/cloud/cloud1/work/ranger/")

# Parámetros de entrada
ksemilla <- c(104729, 100193, 103399, 102253, 104681)

# Parámetros del dataset
karchivo_entrada       <- "~/cloud/cloud1/datasets/adult_extendido.txt"
kcampo_clase           <- "clase"
kvalor_clase_positivo  <- ">50K"
kcampos_a_borrar       <- c( 'fnlwgt' )

# Parámetros del ranger
kformula         <- formula(paste( kcampo_clase, "~ ."))

# Parámetro de la métrica
klift_corte      <-  5000/32561

# Parámetros de la optimización bayesiana
karchivo_salida  <- "~/cloud/cloud1/work/ranger/ranger_bayesiansearch_salida.txt"
karchivo_grid    <- "~/cloud/cloud1/work/ranger/ranger_bayesiansearch_grid.txt"
kbayesian_iniciales  <- 50
kbayesian_total      <- 200

#------------------------------------------------------------------------------
# Cálculo de la función lift

lift = function( vprob, vclase, valor_pos, vcorte ) 
{
  largo  <- length(  vclase  )
  vazar  <- runif( largo )
  tx     <- cbind( vprob ,  vclase, vazar  )
  tx_new <- tx[ order(- vprob , vazar ), ]

  valor_a_buscar <- match( valor_pos,  levels( vclase ) )
  primeros     <- round( largo*vcorte ) 
  
  pos_primeros <- sum( tx_new[ 1:primeros, 2] ==  valor_a_buscar )
  pos_total    <- sum( tx_new[ , 2] ==  valor_a_buscar )

  res  <-  (pos_primeros/ primeros )  /  (  pos_total/largo )

  return( res )   
}
#------------------------------------------------------------------------------

ranger_bayes <- function( pmtry, pmin.node.size, pnum.trees ) 
{

  # Defino los vectores en donde van a quedar los resultados de las 5 iteraciones
  aucs     <- c() 
  lifts    <- c() 
  tiempos  <- c()

  for( s in  1:5 )
  {
    # Divido el dataset en training 70% y testing 30%, usando la libreria caret
    set.seed( ksemilla[s]  )
    inTraining <- createDataPartition( dataset[, kcampo_clase], p = .70, list = FALSE)
    dataset_training <- dataset[  inTraining,]
    dataset_testing  <- dataset[ -inTraining,]

 
    # Generación del modelo
    t0 <-  Sys.time()
    modelo <- ranger( kformula, data = dataset_training, probability=TRUE, 
                      num.trees=pnum.trees,  
                      min.node.size=pmin.node.size, 
                      mtry=pmtry,
                      respect.unordered.factors=TRUE
                    )	
	
    t1 <- Sys.time()
    tiempos[s]  <- as.numeric(  t1 - t0, units = "secs")


    # Genero el vector con la predicción en testing
    testing_prediccion  = predict(  modelo, dataset_testing )
  
    # Apareo el vector anterior con la clase de testing
    pred <- prediction(  testing_prediccion$predictions[,2],   dataset_testing[ , kcampo_clase]  )


    # Calculo el AUC
    testing_auc <- performance( pred, "auc" )
    aucs[s]     <-  testing_auc@y.values[[1]] 

    lifts[s] <-  lift( testing_prediccion$predictions[,2]  ,
                       dataset_testing[ , kcampo_clase],
                       kvalor_clase_positivo, 
                       klift_corte
                     )

    # Remuevo y libero la memoria de los objetos usados recientemente
    rm( inTraining, dataset_training, dataset_testing, modelo, testing_prediccion, pred, testing_auc  )
    gc()

  }


  # Grabo en el archivo de salida
  cat( mean(lifts), mean(aucs), mean(tiempos),  pmtry, pmin.node.size, pnum.trees,  format(Sys.time(), "%Y%m%d %H%M%S"), karchivo_entrada,  "ranger", "bayesiansearch", "montecarlo", "\n", sep="\t", file=karchivo_salida, fill=FALSE, append=TRUE )

  # Grabo en el archivo grid
  cat(  pmtry, "\t", pmin.node.size, "\t", pnum.trees, "\t", mean( lifts), "\n", file=karchivo_grid, fill=FALSE, append=TRUE  )

  list( Score = mean(lifts), Pred = 0 )

}
#------------------------------------------------------------------------------

# Grabo la primer línea del archivo de salida
if( !file.exists( karchivo_salida) )
{
 cat( "lift_mean", "auc_mean", "time_mean", 
      "mtry", "min.node.size", "num.trees",
      "fecha", "entrada", "algoritmo",  "optimizacion", "estimacion", 
      "\n", sep="\t", file=karchivo_salida, fill=FALSE, append=FALSE 
     ) 
}



# Genera la línea con los nombres de campos en el archivo grid si no existe.
# (Es case senstive y "Value" es con la V mayúscula.)
if( !file.exists( karchivo_grid) )
{
  cat(  "pmtry", "\t", "pmin.node.size", "\t", "pnum.trees", "\t",
        "Value",     "\n", 
         file=karchivo_grid, fill=FALSE, append=FALSE 
     ) 
}


init_grid    <- read.table( karchivo_grid, header=TRUE, sep="\t" )
init_points  <- max( kbayesian_iniciales- length( init_grid[,2] ),    0 )
iter_faltan  <- max( kbayesian_total - length( init_grid[,2] ),  0 )

# Leo el dataset
dataset <- read.table( karchivo_entrada, header=TRUE, sep="," )

# Borro los campos que no me interesan
dataset <- dataset[ , !(names(dataset) %in%   kcampos_a_borrar  )    ] 


# Aqui se hace la Optimización Bayesiana
OPT_Res <- BayesianOptimization( ranger_bayes,
           	bounds = list(
                                pmtry              =  c( 2L,   13L),  
                                pmin.node.size     =  c( 1L,  500L), 
                                pnum.trees         =  c( 5L, 5000L)
			      ),
	   init_grid_dt = init_grid, init_points = init_points,  n_iter = iter_faltan,
	   acq = "ucb", kappa = 2.576, eps = 0.01,
	   verbose = TRUE
	   )

print( OPT_Res )


# Limpio la memoria
rm( list=ls() )
gc()

quit( save="no" )
