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
  ns <- NS(id)
  tagList(
    h1("High level activity estimates (principal projection)"),
    fluidRow(
      box(
        title = "Activity Estimates",
        shinycssloaders::withSpinner(
          gt::gt_output(ns("activity"))
        ),
        width = 12
      )
    ),
    fluidRow(
      box(
        title = "A&E Attendances",
        shinycssloaders::withSpinner(
          plotly::plotlyOutput(ns("aae"))
        ),
        width = 4
      ),
      box(
        title = "Inpatient Admissions",
        shinycssloaders::withSpinner(
          plotly::plotlyOutput(ns("ip"))
        ),
        width = 4
      ),
      box(
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
mod_principal_high_level_server <- function(id, data) {
  moduleServer(id, function(input, output, session) {
    ns <- session$ns

    START_YEAR <- 2018
    END_YEAR <- 2029

    fyear_str <- \(y) glue::glue("{y}/{(y + 1) %% 100}")

    summary_data <- reactive({
      d <- data() |>
        dplyr::filter(.data$model_run == 0)

      dplyr::bind_rows(
        aae <- d |>
          dplyr::filter(.data$dataset == "aae") |>
          dplyr::mutate(pod = "A&E Attendances"),
        d |>
          dplyr::filter(.data$dataset == "op", .data$measure == "attendances") |>
          dplyr::mutate(
            dplyr::across(
              .data$pod,
              forcats::fct_recode,
              "OP 1st Attendances" = "op_first",
              "OP Follow up Attendances" = "op_follow-up",
              "OP Procedures" = "op_procedure"
            )
          ),
        d |>
          dplyr::filter(.data$dataset == "ip", .data$measure == "admissions") |>
          dplyr::mutate(
            dplyr::across(
              .data$pod,
              forcats::fct_recode,
              "IP Elective Admissions" = "elective_admission",
              "IP Non-Elective Admissions" = "non-elective_admission",
              "Daycase Admissions" = "elective_daycase"
            )
          )
      ) |>
        dplyr::count(.data$pod, .data$type, wt = .data$value) |>
        dplyr::mutate(
          dplyr::across(.data$pod, forcats::fct_relevel, sort),
          year = ifelse(.data$type == "baseline", START_YEAR, END_YEAR)
        ) |>
        dplyr::select(-.data$type) |>
        tidyr::complete(
          year = seq(START_YEAR, END_YEAR),
          .data$pod
        ) |>
        dplyr::group_by(.data$pod) |>
        dplyr::mutate(
          dplyr::across(n, purrr::compose(as.integer, zoo::na.approx)),
          fyear = fyear_str(.data$year)
        ) |>
        dplyr::ungroup()
    })

    output$activity <- gt::render_gt({
      summary_data() |>
        dplyr::select(-.data$year) |>
        tidyr::pivot_wider(names_from = .data$fyear, values_from = .data$n) |>
        gt::gt() |>
        gt::cols_align(
          align = "left",
          columns = "pod"
        ) |>
        gt::cols_label(
          "pod" = ""
        ) |>
        gt::fmt_integer(tidyselect::matches("\\d{4}/\\d{2}"))
    })

    plot_fn <- function(data, activity_type) {
      d <- data |>
        dplyr::filter(.data$pod |> stringr::str_starts(activity_type))

      p <- d |>
        ggplot2::ggplot(aes(.data$year, .data$n, colour = .data$pod)) +
        ggplot2::geom_line() +
        ggplot2::geom_point() +
        ggplot2::scale_x_continuous(
          labels = fyear_str,
          breaks = seq(START_YEAR, END_YEAR, 2)
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
      summary_data() |>
        plot_fn("A&E")
    })

    output$ip <- plotly::renderPlotly({
      summary_data() |>
        plot_fn("IP")
    })

    output$op <- plotly::renderPlotly({
      summary_data() |>
        plot_fn("OP")
    })
  })
}