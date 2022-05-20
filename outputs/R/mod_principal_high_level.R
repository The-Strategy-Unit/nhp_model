#' principal_high_level UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_principal_high_level_ui <- function(id) {
  ns <- shiny::NS(id)
  shiny::tagList(
    shiny::h1("High level activity estimates (principal projection)"),
    shiny::fluidRow(
      bs4Dash::box(
        title = "Activity Estimates",
        shinycssloaders::withSpinner(
          gt::gt_output(ns("activity"))
        ),
        width = 12
      )
    ),
    shiny::fluidRow(
      bs4Dash::box(
        title = "A&E Attendances",
        shinycssloaders::withSpinner(
          plotly::plotlyOutput(ns("aae"))
        ),
        width = 4
      ),
      bs4Dash::box(
        title = "Inpatient Admissions",
        shinycssloaders::withSpinner(
          plotly::plotlyOutput(ns("ip"))
        ),
        width = 4
      ),
      bs4Dash::box(
        title = "Outpatient Attendances",
        shinycssloaders::withSpinner(
          plotly::plotlyOutput(ns("op"))
        ),
        width = 4
      )
    )
  )
}

#' principal_high_level Server Functions
#'
#' @noRd
mod_principal_high_level_server <- function(id, selected_model_run, data_cache) {
  shiny::moduleServer(id, function(input, output, session) {
    fyear_str <- function(y) glue::glue("{y}/{(y + 1) %% 100}")

    pods <- get_activity_type_pod_measure_options() |>
      dplyr::filter(.data$activity_type != "aae") |>
      distinct(.data$activity_type, .data$pod, .data$pod_name) |>
      dplyr::bind_rows(data.frame(activity_type = "aae", pod = "aae", pod_name = "A&E Attendance")) |>
      dplyr::mutate(dplyr::across(.data$pod_name, forcats::fct_inorder))

    summary_data <- reactive({
      c(ds, sc, cd) %<-% selected_model_run()

      params <- cosmos_get_params(ds, sc, cd)

      c(start_year, end_year) %<-% params[["demographic_factors"]][c("start_year", "end_year")]

      cosmos_get_principal_highlevel(ds, sc, cd) |>
        dplyr::mutate(year = ifelse(.data$model_run == "baseline", start_year, end_year)) |>
        dplyr::select(-.data$model_run) |>
        tidyr::complete(
          year = seq(start_year, end_year),
          .data$pod
        ) |>
        dplyr::inner_join(pods, by = "pod") |>
        dplyr::group_by(.data$activity_type, pod = .data$pod_name) |>
        dplyr::mutate(
          dplyr::across(n, purrr::compose(as.integer, zoo::na.approx)),
          fyear = fyear_str(.data$year)
        ) |>
        dplyr::ungroup()
    }) |>
      shiny::bindCache(selected_model_run(), cache = data_cache)

    output$activity <- gt::render_gt({
      summary_data() |>
        dplyr::select(-.data$activity_type, -.data$year) |>
        tidyr::pivot_wider(names_from = .data$fyear, values_from = .data$n) |>
        gt::gt() |>
        gt::cols_align(
          align = "left",
          columns = "pod"
        ) |>
        gt::cols_label(
          "pod" = ""
        ) |>
        gt::fmt_integer(tidyselect::matches("\\d{4}/\\d{2}")) |>
        gt_theme()
    })

    plot_fn <- function(data, activity_type) {
      c(start_year, end_year) %<-% range(data$year)

      d <- data |>
        dplyr::filter(.data$activity_type == .env$activity_type)

      p <- d |>
        ggplot2::ggplot(aes(.data$year, .data$n, colour = .data$pod)) +
        ggplot2::geom_line() +
        ggplot2::geom_point() +
        ggplot2::scale_x_continuous(
          labels = fyear_str,
          breaks = seq(start_year, end_year, 2)
        ) +
        ggplot2::scale_y_continuous(
          labels = scales::comma
        ) +
        ggplot2::expand_limits(y = 0) +
        ggplot2::labs(x = NULL, y = NULL, colour = NULL)

      plotly::ggplotly(p) %>%
        plotly::layout(legend = list(
          orientation = "h"
        ))
    }

    output$aae <- plotly::renderPlotly({
      plot_fn(summary_data(), "aae")
    })

    output$ip <- plotly::renderPlotly({
      plot_fn(summary_data(), "ip")
    })

    output$op <- plotly::renderPlotly({
      plot_fn(summary_data(), "op")
    })
  })
}