library(shiny)

# Define UI ----
ui <- fluidPage(
  titlePanel("My Shiny App"),
  sidebarLayout(
    sidebarPanel(
      h1("Installation"),
      p("Shiny is available on CRAN, soi you can install it in the usual way from your R consol:"),
      code("install.packages('rshiny')"),
      br(),
      br(),
      br(),
      br(),
      img(src = "rstudio.png", height = 70, width = 200),
      br(),
      "Shiny is a product of ", 
      span("RStudio", style = "color:blue")
    ),
    mainPanel(
      # The img function looks for your image file in a specific place. 
      # Your file must be in a folder named www in the same directory as 
      # the app.R script. Shiny treats this directory in a special way. 
      #Shiny will share any file placed here with your user’s web browser, 
      #which makes www a great place to put images, style sheets, 
      #and other things the browser will need to build the web components 
      #of your Shiny app.
      h1("Introducing Shiny"),
      p("Shiny is a new package from RStudio that makes it ", 
        em("incredibly easy "), 
        "to build interactive web applications with R."),
      br(),
      p("For an introduction and live examples, visit the ",
        a("Shiny homepage.", 
          href = "http://shiny.rstudio.com")),
      br(),
      h2("Features"),
      p("- Build useful web applications with only a few lines of code—no JavaScript required."),
      p("- Shiny applications are automatically 'live' in the same way that ", 
        strong("spreadsheets"),
        " are live. Outputs change instantly as users modify inputs, without requiring a reload of the browser.")
    )
  )
)


# Define server logic ----
server <- function(input, output) {
  
}

# Run the app ----
shinyApp(ui = ui, server = server)