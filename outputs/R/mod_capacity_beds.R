#' capacity_beds UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_capacity_beds_ui <- function(id) {
  ns <- shiny::NS(id)
  tagList(
    shiny::numericInput(ns("occupancy_rate"), "Occupancy Rate (%)", 85, 0, 100, 1),
    shiny::fluidRow(
      bs4Dash::column(
        width = 9,
        shinycssloaders::withSpinner(
          plotly::plotlyOutput(ns("available_plot"), height = "800px"),
        )
      ),
      bs4Dash::column(
        width = 3,
        shinycssloaders::withSpinner(
          gt::gt_output(ns("available_table"))
        )
      )
    )
  )
}

#' capacity_beds Server Functions
#'
#' @noRd
mod_capacity_beds_server <- function(id, selected_model_run, data_cache) {
  specialties <- readr::read_csv(app_sys("specialties.csv"), col_types = "cc__c")

  get_beds_data <- function(ds, id) {
    kh03 <- readr::read_csv(app_sys("data", "kh03", paste0(ds, ".csv")), lazy = FALSE, col_types = "ccdd")

    cosmos_get_mainspef_agg(id) |>
      dplyr::group_by(.data$mainspef) |>
      dplyr::summarise(dplyr::across(where(is.numeric), sum)) |>
      dplyr::inner_join(kh03, by = c("mainspef" = "specialty_code"))
  }

  get_new_available_beds <- function(data, occupancy_rate) {
    data |>
      dplyr::filter(.data$baseline > 0, .data$principal > 0) |>
      dplyr::mutate(
        new_available = (.data$principal / .data$baseline) * .data$occupied * (1 / .env$occupancy_rate)
      ) |>
      dplyr::inner_join(specialties, by = c("mainspef" = "code")) |>
      dplyr::mutate(dplyr::across(.data$specialty_group, snakecase::to_title_case))
  }

  get_available_table <- function(data) {
    data |>
      dplyr::select(
        .data$specialty_group,
        specialty = .data$description,
        old_beds = .data$available,
        new_beds = .data$new_available
      ) |>
      dplyr::arrange(desc(new_beds), desc(old_beds)) |>
      dplyr::filter(new_beds > 0.5) |>
      gt::gt(rowname_col = "specialty", groupname_col = "specialty_group") |>
      gt::fmt_integer(c("old_beds", "new_beds")) |>
      gt::summary_rows(
        groups = TRUE,
        columns = c("old_beds", "new_beds"),
        fns = list(total = "sum"),
        formatter = gt::fmt_integer
      ) |>
      gt::cols_label(
        old_beds = "Baseline",
        new_beds = "Principal"
      ) |>
      gt::tab_spanner("Number of Beds", c("old_beds", "new_beds")) |>
      gt::tab_options(
        row_group.background.color = "#686f73",
        summary_row.background.color = "#b2b7b9"
      )
  }

  get_available_plot <- function(data) {
    data |>
      dplyr::mutate(across(.data$description, forcats::fct_reorder, .data$new_available)) |>
      dplyr::filter(.data$new_available > 5) |>
      ggplot2::ggplot(ggplot2::aes(.data$new_available, .data$description, fill = .data$type)) +
      ggplot2::geom_col() +
      ggplot2::geom_errorbar(ggplot2::aes(xmin = .data$available, xmax = .data$available)) +
      ggplot2::facet_wrap(ggplot2::vars(.data$specialty_group), scales = "free")
  }

  moduleServer(id, function(input, output, session) {
    beds_data <- shiny::reactive({
      c(ds, sc, mr, id) %<-% selected_model_run() # nolint
      get_beds_data(ds, id) # nolint
    }) |>
      shiny::bindCache(selected_model_run(), cache = data_cache)

    new_beds <- shiny::reactive({
      occupancy_rate <- shiny::req(input$occupancy_rate) / 100

      get_new_available_beds(beds_data(), occupancy_rate)
    })

    output$available_table <- gt::render_gt({
      get_available_table(new_beds())
    })

    output$available_plot <- plotly::renderPlotly({
      new_beds() |>
        dplyr::filter(.data$specialty_group == "General and Acute") |>
        get_available_plot() |>
        plotly::ggplotly() |>
        plotly::layout(legend = list(
          orientation = "h"
        ))
    })
  })
}

## To be copied in the UI

## To be copied in the server
# mod_capacity_beds_server("capacity_beds_1")
