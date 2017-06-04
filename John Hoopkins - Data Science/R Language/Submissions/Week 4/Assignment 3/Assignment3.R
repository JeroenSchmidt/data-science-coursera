## Global Variables
heartAttack.MortalityRate <- "Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack" 
pneumonia.MortalityRate <-  "Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia"  
heartFailure.MortalityRate <- "Hospital.30.Day.Death..Mortality..Rates.from.Heart.Failure"  


check.arg.outcome <- function(outcome){
  if (outcome == "heart attack"){
    return(heartAttack.MortalityRate)
  }else if (outcome == "pneumonia"){
    return(pneumonia.MortalityRate)
  }else if (outcome == "heart failure"){
    return(heartFailure.MortalityRate) 
  }else{
    stop("invalid outcome")
  }
}

## Find hospital in specified state with lowest mortality 
best <- function(state,outcome){
  
  ## Read outcome data
  csvDF <- read.csv("Data/outcome-of-care-measures.csv", colClasses = "character")
  
  ## Check input arguments 
  # Check that outcome is valid
  outcome.col <- check.arg.outcome(outcome)

  # Check that state is in csv file
  states <- unique(csvDF["State"])
  if (!any(state == states)){
    stop("invalid state")
  }
  
  ## return the hospital name in the state with lowest 30-day death rate
  # select relevant rows
  csvDF <- csvDF[c("Hospital.Name","State",outcome.col)]
  
  # filter out bad values
  filterOut.BadValue <- csvDF[outcome.col] != "Not Available"
  csvDF <- csvDF[filterOut.BadValue,]

  # filter out non-relevant states
  keep.state <- csvDF["State"] == state
  csvDF <- csvDF[keep.state,]
  
  # convert outcome row
  csvDF[outcome.col] <- as.numeric(csvDF[,outcome.col]) 
  
  # determine best hospital 
  bestRowIndex <- which.min(csvDF[,outcome.col])
  bestHospital <- csvDF[bestRowIndex,"Hospital.Name"]
  
  return(bestHospital)
}

## Ranking Hospitals by outcome in a state
##----------------------- NOT FINISHED
rankhospital <- function(state,outcome,num){
  ## Read outcome data
  csvDF <- read.csv("Data/outcome-of-care-measures.csv", colClasses = "character")
  
  ## Check input arguments 
  # Check that outcome is valid
  outcome.col <- check.arg.outcome(outcome)
  
  # Check that state is in csv file
  states <- unique(csvDF["State"])
  if (!any(state == states)){
    stop("invalid state")
  }
  
  ## Return hospital name in that state with the given rank ## 30-day death rate
  # convert outcome row
  csvDF <- csvDF[c("Hospital.Name","State",outcome.col)]
  csvDF[outcome.col] <- as.numeric(csvDF[,outcome.col]) 
  
  # filter out non-relevant states
  keep.state <- csvDF["State"] == state
  csvDF <- csvDF[keep.state,]
  orderedDF <- csvDF[order(csvDF[outcome.col],rev(csvDF["Hospital.Name"]), na.last = NA),]
  
  if (class(num) != "numeric"){
    if (num == "best"){
      num <- 1
    }else if (num == "worst"){
      num <- nrow(orderedDF)
    }
  }
  
  row.count <- nrow(orderedDF)
  if (row.count<num){
    return(NA)
  }
  
  orderedDF$Rank <- 1:row.count
  return(orderedDF[(orderedDF$Rank == num),"Hospital.Name"])
}

rankall <- function(outcome, num = "best") {
  ## Read outcome data
  csvDF <- read.csv("Data/outcome-of-care-measures.csv", colClasses = "character")
  
  ## Check input arguments 
  # Check that outcome is valid
  outcome.col <- check.arg.outcome(outcome)
  
  ## For each state, find the hospital of the given rank
  ## Return hospital name in that state with the given rank ## 30-day death rate
  # convert outcome row
  csvDF <- csvDF[c("Hospital.Name","State",outcome.col)]
  csvDF[outcome.col] <- as.numeric(csvDF[,outcome.col]) 
  orderedDF <- csvDF[order(csvDF[outcome.col],csvDF["Hospital.Name"], na.last = NA),]
  
  #x <- split(orderedDF,orderedDF$State)
  #return(x$NV)
  
  ranks <- lapply(split(orderedDF,orderedDF$State),function(x){
    
    rank <- num  
    if (class(num) == "character"){
      if (num == "best"){
        rank <- 1
      }else if (num == "worst"){
        rank <- dim(x)[1]
      }
    }
    x[rank,"Hospital.Name"]#["Hospital.Name"]
  })
  
  output <- data.frame(hospital=unlist(x,use.names = FALSE),state=names(ranks),row.names=names(ranks))
  
  return(output)

}

