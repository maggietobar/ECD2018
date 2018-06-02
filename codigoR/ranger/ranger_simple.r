# RANGER: cálculo correcto de AUC  y LIFT  haciendo Repeated random sub-sampling validation

# Limpio la memoria
rm( list=ls() )
gc()

# source( "~/cloud/cloud1/codigoR/ranger/ranger_simple.R" )

library(ranger)
library(caret)
library(ROCR)
library(randomForest)  #solo se usa para imputar nulos

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

# Establezco la carpeta de trabajo
#setwd( "~/cloud/cloud1/work/ranger/")

# Parámetros de entrada
ksemilla <- c(104729, 100193, 103399, 102253, 104681)

# Parámetros del dataset
karchivo_entrada       <- "~/cloud/cloud1/datasets/adult_extendido.txt"
kcampo_clase           <- "clase"
kvalor_clase_positivo  <- ">50K"
kcampos_a_borrar       <- c( 'fnlwgt' )

# Parámetros de ranger
kformula         <- formula(paste( kcampo_clase, "~ ."))

# Parámetros de la metrica
klift_corte      <-  5000/32561

# Cargo el dataset
dataset <- read.table( karchivo_entrada, header=TRUE, sep="," )

# Borro los campos que no me interesan
dataset <- dataset[ , !(names(dataset) %in%   kcampos_a_borrar  )    ] 

# Imputo los nulos, ya que ranger no acepta nulos
dataset <-  na.roughfix( dataset )

# Inicializo los vectores donde quedan los resultados
aucs    <- c() 
lifts   <- c()
tiempos <- c()

for( s in  1:5 )
{
  # Divido el dataset en training 70% y testing 30%, usando la libreria caret
  set.seed( ksemilla[s]  )
  inTraining <- createDataPartition( dataset[, kcampo_clase], p = .70, list = FALSE)
  dataset_training <- dataset[  inTraining,]
  dataset_testing  <- dataset[ -inTraining,]


  # Generación del modelo
  t0 <-  Sys.time()
  modelo <- ranger( kformula, data = dataset_training, probability=TRUE, respect.unordered.factors=TRUE )  
  t1 <- Sys.time()
  tiempos[s]  <- as.numeric(  t1 - t0, units = "secs")


  # Genero el vector con la predicción en testing
  testing_prediccion  = predict(  modelo, dataset_testing )

  # Apareo el vector anterior con la clase de testing
  pred <- prediction(  testing_prediccion$predictions[,2],   dataset_testing[ , kcampo_clase]  )
  
  # Calculo el AUC y LIFT
  testing_auc <- performance( pred, "auc" )
  aucs[s]     <-  testing_auc@y.values[[1]] 

  lifts[s] <-  lift( testing_prediccion$predictions[,2]  ,
                     dataset_testing[ , kcampo_clase],
                     kvalor_clase_positivo, 
                     klift_corte
                   )

}


print( tiempos )
print( aucs )
print( mean( aucs) )
print( lifts )
print( mean( lifts) )
