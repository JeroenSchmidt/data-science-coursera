## Global Variables
heartAttack.MortalityRate = "Lower.Mortality.Estimate...Hospital.30.Day.Death..Mortality..Rates.from.Heart.Attack"
pneumonia.MortalityRate = "Lower.Mortality.Estimate...Hospital.30.Day.Death..Mortality..Rates.from.Pneumonia"  

## Find hospital in specified state with lowest mortality 
best <- function(state,outcome){
  
  ## Read outcome data
  csvDF <- read.csv("Data/outcome-of-care-measures.csv", colClasses = "character")
  
  ## Check input arguments 
  # Check that outcome is valid
  if (outcome == "heart attack"){
    outcome.col <- heartAttack.MortalityRate
  }else if (outcome == "pneumonia"){
    outcome.col <- pneumonia.MortalityRate
  }else{
    stop("invalid outcome")
  }

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
  if (outcome == "heart attack"){
    outcome.col <- heartAttack.MortalityRate
  }else if (outcome == "pneumonia"){
    outcome.col <- pneumonia.MortalityRate
  }else{
    stop("invalid outcome")
  }
  
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
  csvDF <- csvDF[order(csvDF[outcome.col]),]
  return(csvDF)
}
