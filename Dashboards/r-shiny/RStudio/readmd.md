# RStudio Tutorials on RShiny

Tutorials from https://shiny.rstudio.com/tutorial/

## Lesson 1

## Lesson 2

## Lesson 3

## Lesson 4
In this lesson, you created your first reactive Shiny app. Along the way, you learned to

    * use an `*Output` function in the `ui` to place reactive objects in your Shiny app,
    * use a `render*` function in the `server` to tell Shiny how to build your objects,
    * surround R expressions by curly braces, `{}`, in each `render*` function,
    * save your `render*` expressions in the output list, with one entry for each reactive object in your app, and
    * create reactivity by including an `input` value in a `render*` expression.

If you follow these rules, Shiny will automatically make your objects reactive.

In Lesson 5 you will create a more sophisticated reactive app that relies on R scripts and external data.

## Lesson 5
![Data DL](https://shiny.rstudio.com/tutorial/written-tutorial/lesson5/census-app/data/counties.rds)

You can create more complicated Shiny apps by loading R Scripts, packages, and data sets.

Keep in mind:

    * The directory that app.R appears in will become the working directory of the Shiny app
    * Shiny will run code placed at the start of app.R, before the server function, only once during the life of the app.
    * Shiny will run code placed inside server function multiple times, which can slow down the app.

You also learned that switch is a useful companion to multiple choice Shiny widgets. Use switch to change the values of a widget into R expressions.

As your apps become more complex, they can become inefficient and slow. Lesson 6 will show you how to build fast, modular apps with reactive expressions.