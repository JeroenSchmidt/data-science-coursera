pollutantmean <- function(directory, pollutant = "sulfate", ids = 1:332){
  ## 1) 'directory' is a character vector of length 1 indicating the location of the CSV files
  
  ## 2) 'pollutant' is a character vector of length 1 indicating the name of the pollutant for 
  ## which we weill calculate the mean; either "sulfate" or "nitrate"
  
  ## 3) 'ID' is na integer vector indicating the moitor ID numbrs to be used
  
  globalNodValues <- numeric(0)
  
  for (id in ids){
    fileID <- toString(id)
    len <- nchar(fileID)

    if (len == 2){
      fileID <- paste("0",fileID,sep = "")      
    }else if(len == 1){
      fileID <- paste("00",fileID,sep = "")
    }
    
    fileName <- paste(fileID,".csv",sep = "") 
    
    filePath <- file.path(".",directory,fileName)
    
    if (file.exists(filePath)){
      dFrame <- read.csv(filePath)
      completeLogic <- (!is.na(dFrame["sulfate"]) & !is.na(dFrame["nitrate"])) 
      
      filteredResults <- dFrame[completeLogic,]
      
      globalNodValues <- c(globalNodValues,filteredResults[[pollutant]])
    }
  }
  globalMean <- mean(globalNodValues)
}


complete <- function(directory, ids = 1:333){
  ## 'direcotry' is a character vector of length 1 indicating the location of the CSV file
  
  ## 'id' us an integer vector indicating the moitor ID numbers to be used
  
  result <- data.frame(id = ids, nods = numeric(length(ids)))
  rownames(result) <- result$id
  
  for (id in ids){
    fileID <- toString(id)
    len <- nchar(fileID)
    
    if (len == 2){
      fileID <- paste("0",fileID,sep = "")      
    }else if(len == 1){
      fileID <- paste("00",fileID,sep = "")
    }
    
    fileName <- paste(fileID,".csv",sep = "") 
    
    filePath <- file.path(".",directory,fileName)
    
    if (file.exists(filePath)){
      dFrame <- read.csv(filePath)
      completeLogic <- (!is.na(dFrame["sulfate"]) & !is.na(dFrame["nitrate"])) 

      completeNodes <- sum(completeLogic)
      
      result[toString(id),]$nods <- completeNodes
    }
  }
  return(result)
}

corr <- function(directory,threshold = 0){
  ## 'directory' is a character vector of length 1 indicating the location of the CSV files
  
  ## 'threshold' is a numeric vector of length 1 indicating the number of 
  ## completely observed observations (on all variables) required to compute the correlation between 
  ## nitirate and sulfate; default is 0
  
  nodCount <- complete(directory,ids = 1:333)
  
  thresholdID <- nodCount[(nodCount$nods >= threshold),]$id
  
  if (length(thresholdID) < 1) {
    return(numeric(0))
  }
  
  result <- data.frame(id = thresholdID, corr = numeric(length(thresholdID)))
  rownames(result) <- result$id
  
  for (id in thresholdID){
    fileID <- toString(id)
    len <- nchar(fileID)
    
    if (len == 2){
      fileID <- paste("0",fileID,sep = "")      
    }else if(len == 1){
      fileID <- paste("00",fileID,sep = "")
    }
    
    fileName <- paste(fileID,".csv",sep = "") 
    
    filePath <- file.path(".",directory,fileName)
    
    if (file.exists(filePath)){
      dFrame <- read.csv(filePath)
      
      completeLogic <- (!is.na(dFrame["sulfate"]) & !is.na(dFrame["nitrate"])) 
      
      filterDF <- dFrame[completeLogic,]
      
      corrolation <- cor(filterDF$nitrate,filterDF$sulfate)
      
      result[toString(id),]$corr <- corrolation
    }
  }
  return(result)
}
