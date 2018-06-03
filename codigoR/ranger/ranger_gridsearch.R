# Ranger 
# Montecarlo estimation
# Búsqueda de parámetros óptimos con Grid Search
# Optimizo  Lift


# Limpio la memoria
rm( list=ls() )
gc()


# source( "~/cloud/cloud1/codigoR/ranger/ranger_gridsearch.R" )

library(ranger)
library(caret)
library(ROCR)
library(randomForest)  #solo se usa para imputar nulos


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


# Parámetro de la metrica
klift_corte      <-  5000/32561

# Parámetro de grid search
karchivo_salida  <- "~/cloud/cloud1/work/ranger/ranger_gridsearch_salida.txt"


#------------------------------------------------------------------------------
# Cálculo de la funcion lift

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

  lift_calculado  <-  (pos_primeros/ primeros )  /  (  pos_total/largo )

  return( lift_calculado )   
}
#------------------------------------------------------------------------------

# Levanto el dataset
dataset <- read.table( karchivo_entrada, header=TRUE, sep="," )

# Borro los campos que no me interesan
dataset <- dataset[ , !(names(dataset) %in%   kcampos_a_borrar  )    ] 

# Imputo los nulos, ya que ranger no acepta nulos
dataset <-  na.roughfix( dataset )

# Grabo la primer linea del archivo de salida
if( !file.exists( karchivo_salida) )
{
 cat( "lift_mean", "auc_mean", "time_mean", 
      "auc1", "auc2", "auc3", "auc4", "auc5",
      "num.trees", "vmin.node.size", "mtry",  "splitrule",
      "fecha", "entrada", "algoritmo",  "optimizacion", "estimacion", 
      "\n", sep="\t", file=karchivo_salida, fill=FALSE, append=FALSE 
     ) 
}

lineas_archivo <-  length( readLines(karchivo_salida) )  - 1

linea <- 1


for( vsplitrule  in  c( "gini", "extratrees" ) )
{
for( vmtry   in  c(  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 )  )
{
for( vmin.node.size  in  c( 500, 300, 200, 100, 50, 30, 20, 10, 5, 1)  )
{
for( vnum.trees   in  c( 5, 10, 20, 50, 100, 200, 500, 800, 1000, 1500, 2000, 5000) )
{

  # Inicializo los vectores donde quedan los resultados
  aucs    <- c() 
  lifts   <- c()
  tiempos <- c()

  if( linea > lineas_archivo )
  {
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
                        num.trees=vnum.trees,  
                        min.node.size=vmin.node.size, 
                        mtry=vmtry, 
                        splitrule=vsplitrule,
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

    # Grabo los resultados de esta corrida de ranger
    cat( mean(lifts), mean(aucs), mean(tiempos), aucs ,
         vnum.trees , vmin.node.size, vmtry, vsplitrule, 
         format(Sys.time(), "%Y%m%d %H%M%S"), karchivo_entrada,  "ranger",  "gridsearch", "montecarlo",
         "\n", sep="\t", file=karchivo_salida, fill=FALSE, append=TRUE )

  }
  linea <- linea + 1

}
}
}
}




# Limpio la memoria
rm( list=ls() )
gc()


quit( save="no" )
