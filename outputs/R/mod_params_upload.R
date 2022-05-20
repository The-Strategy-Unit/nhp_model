#' params_upload UI Function
#'
#' @description A shiny Module.
#'
#' @param id,input,output,session Internal parameters for {shiny}.
#'
#' @noRd
#'
#' @importFrom shiny NS tagList
mod_params_upload_ui <- function(id) {
  ns <- shiny::NS(id)

  card_file_selection <- shiny::tagList(
    shiny::selectInput(ns("dataset"), "Dataset", choices = NULL),
    shiny::selectInput(
      ns("demographics_file"),
      "Demographics File",
      choices = c("default" = "demographic_factors.csv")
    ),
    shiny::textInput(ns("scenario_name"), "Scenario Name"),
    shinyjs::disabled(
      shiny::fileInput(ns("params_upload"), "Upload Params Excel File", accept = ".xlsx")
    ),
    shiny::textOutput(ns("status"))
  )

  tab_run_settings <- shiny::tagList(
    shiny::textInput(ns("start_year"), "Start Year"),
    shiny::textInput(ns("end_year"), "End Year"),
    shiny::textInput(ns("model_iterations"), "Model Iterations"),
    shiny::actionButton(ns("submit_model_run"), "Submit Model Run"),
    shiny::downloadButton(ns("download_params"), "Download params.json")
  )

  tab_demographics <- shiny::tagList(
    shiny::h4("Population Model Variant Probabilities"),
    shiny::textInput(ns("demo_principal"), "Principal"),
    shiny::textInput(ns("demo_high_migration"), "High Migration")
  )

  tab_hsa <- shiny::tagList(
    shiny::textInput(ns("hsa_lo"), "Low Estimate (5th Percentile)"),
    shiny::textInput(ns("hsa_hi"), "High Estimate (95th Percentile)"),
  )

  tab_ip <- shiny::tagList(
    bs4Dash::box(
      title = "Activity Avoidance",
      width = 12,
      collapsible = FALSE,
      shiny::selectInput(ns("ip_am_a"), "Change Factor", choices = NULL),
      shiny::fluidRow(
        col_6(shiny::textInput(ns("ip_am_a_lo"), "Low Estimate (5th Percentile)")),
        col_6(shiny::textInput(ns("ip_am_a_hi"), "High Estimate (85th Percentile)"))
      )
    ),
    bs4Dash::box(
      title = "Type Convesion (BADS)",
      width = 12,
      collapsible = FALSE,
      shiny::selectInput(ns("ip_am_tc_bads"), "Change Factor", choices = NULL),
      shiny::fluidRow(
        col_3(shiny::textInput(ns("ip_am_tc_bads_br"), "Baseline Rate")),
        col_3(shiny::textInput(ns("ip_am_tc_bads_split"), "Split (OP:DC)")),
        col_3(shiny::textInput(ns("ip_am_tc_bads_lo"), "Low Estimate (5th Percentile)")),
        col_3(shiny::textInput(ns("ip_am_tc_bads_hi"), "High Estimate (85th Percentile)"))
      )
    ),
    bs4Dash::box(
      title = "Efficiencies",
      width = 12,
      collapsible = FALSE,
      shiny::selectInput(ns("ip_am_e"), "Change Factor", choices = NULL),
      shiny::fluidRow(
        col_6(shiny::textInput(ns("ip_am_e_lo"), "Low Estimate (5th Percentile)")),
        col_6(shiny::textInput(ns("ip_am_e_hi"), "High Estimate (85th Percentile)"))
      )
    )
  )

  table_fn <- function(n, a, b) {
    nfn <- \(x) ns(paste(n, x, sep = "_"))
    shiny::tags$table(
      shiny::tags$tr(
        shiny::tags$th("Age Group"),
        shiny::tags$th("Specialty Group"),
        shiny::tags$th("Low Estimate (5th Percentile)"),
        shiny::tags$th("High Estimate (95th Percentile)"),
      ),
      shiny::tags$tr(
        shiny::tags$td("Child", rowspan = 2),
        shiny::tags$td(a),
        shiny::tags$td(shiny::textInput(nfn("lo_c1"), NULL)),
        shiny::tags$td(shiny::textInput(nfn("hi_c1"), NULL))
      ),
      shiny::tags$tr(
        shiny::tags$td(b),
        shiny::tags$td(shiny::textInput(nfn("lo_c2"), NULL)),
        shiny::tags$td(shiny::textInput(nfn("hi_c2"), NULL))
      ),
      shiny::tags$tr(
        shiny::tags$td("Adult", rowspan = 2),
        shiny::tags$td(a),
        shiny::tags$td(shiny::textInput(nfn("lo_a1"), NULL)),
        shiny::tags$td(shiny::textInput(nfn("hi_a1"), NULL))
      ),
      shiny::tags$tr(
        shiny::tags$td(b),
        shiny::tags$td(shiny::textInput(nfn("lo_a2"), NULL)),
        shiny::tags$td(shiny::textInput(nfn("hi_a2"), NULL))
      )
    )
  }

  tab_op <- tagList(
    bs4Dash::box(
      title = "Activity Avoidance",
      width = 12,
      collapsible = FALSE,
      shiny::selectInput(ns("op_am_a"), "Change Factor", choices = NULL),
      table_fn("op_am_a", "surgical", "non-surgical")
    ),
    bs4Dash::box(
      title = "Type Convesion (F2F to Telephone)",
      width = 12,
      collapsible = FALSE,
      shiny::selectInput(ns("op_am_tc_tele"), "Change Factor", choices = NULL),
      table_fn("op_am_tc_tele", "non-surgical", "surgical")
    )
  )

  tab_aae <- shiny::tagList(
    bs4Dash::box(
      title = "Activity Avoidance",
      width = 12,
      collapsible = FALSE,
      shiny::selectInput(ns("aae_am_a"), "Change Factor", choices = NULL),
      table_fn("aae_am_a", "ambulance", "walk-in")
    )
  )

  tagList(
    bs4Dash::box(
      width = 12,
      title = "Upload File",
      card_file_selection
    ),
    shinyjs::disabled(
      bs4Dash::tabBox(
        id = "params",
        width = 12,
        collapsible = FALSE,
        type = "pills",
        shiny::tabPanel("Run Settings", tab_run_settings),
        shiny::tabPanel("Demographics", tab_demographics),
        shiny::tabPanel("Health Status Adjustment", tab_hsa),
        shiny::tabPanel("IP Activity Mitigators", tab_ip),
        shiny::tabPanel("OP Activity Mitigators", tab_op),
        shiny::tabPanel("AAE Activity Mitigators", tab_aae)
      )
    )
  )
}

#' params_upload Server Functions
#'
#' @noRd
mod_params_upload_server <- function(id) {
  moduleServer(id, function(input, output, session) {
    shiny::observe({
      shiny::updateSelectInput(session, "dataset", choices = c("synthetic", "RL4"))
    })

    status <- shiny::reactiveVal("waiting")

    shiny::observe({
      valid <- c("dataset", "demographics_file", "scenario_name") |>
        purrr::every(~ shiny::isTruthy(input[[.x]]))

      shinyjs::toggleState("params_upload", valid)
      shinyjs::toggleState("submit_model_run", valid)
      shinyjs::toggleState("download_params", valid)
      shinyjs::toggleState("ip_am_a", valid)
      shinyjs::toggleState("op_am_a", valid)
      shinyjs::toggleState("aae_am_a", valid)
      shinyjs::toggleState("ip_am_tc_bads", valid)
      shinyjs::toggleState("op_am_tc_tele", valid)
      shinyjs::toggleState("ip_am_e", valid)

      shiny::updateActionButton(session, "submit_model_run", "Submit Model Run")
    })

    params <- shiny::reactive({
      dataset <- input$dataset
      demographics_file <- input$demographics_file
      scenario_name <- input$scenario_name

      file <- shiny::req(input$params_upload)
      ext <- tools::file_ext(file$datapath)
      shiny::validate(
        shiny::need(ext == "xlsx", "Please upload an xlsx file"),
        shiny::need(shiny::isTruthy(dataset), "Select a dataset"),
        shiny::need(shiny::isTruthy(demographics_file), "Select a demographics file"),
        shiny::need(shiny::isTruthy(scenario_name), "Enter a scenario name")
      )

      status("processing file...")
      on.exit({
        status("file loaded, displaying loaded params")
      })
      process_param_file(file$datapath, dataset, demographics_file, scenario_name)
    }) |>
      shiny::bindEvent(input$params_upload) |>
      shiny::debounce(1000)

    shiny::observeEvent(params(), {
      p <- params()

      # run settings
      shiny::updateTextInput(session, "start_year", value = p$start_year)
      shiny::updateTextInput(session, "end_year", value = p$end_year)
      shiny::updateTextInput(session, "model_iterations", value = p$model_runs)

      # demographics
      vp <- p$demographic_factors$variant_probabilities
      shiny::updateTextInput(session, "demo_principal", value = vp[["principal"]])
      shiny::updateTextInput(session, "demo_high_migration", value = vp[["high migration"]])

      # health status adjustment
      shiny::updateTextInput(session, "hsa_lo", value = p$health_status_adjustment[[1]])
      shiny::updateTextInput(session, "hsa_hi", value = p$health_status_adjustment[[2]])

      # admission avoidance
      ## ip
      shiny::updateSelectInput(session, "ip_am_a", choices = names(p$strategy_params$admission_avoidance))
      shiny::updateSelectInput(
        session,
        "op_am_a",
        choices = p$outpatient_factors |>
          names() |>
          purrr::discard(~ .x == "convert_to_tele")
      )
      shiny::updateSelectInput(session, "aae_am_a", choices = names(p$aae_factors))

      # type conversion
      ## ip
      ### bads
      tc_bads <- p$strategy_params$los_reduction |>
        purrr::keep(~ .x$type == "bads") |>
        names()
      shiny::updateSelectInput(session, "ip_am_tc_bads", choices = tc_bads)
      ## op
      ### convert to tele
      shiny::updateSelectInput(
        session,
        "op_am_tc_tele",
        choices = p$outpatient_factors |>
          names() |>
          purrr::keep(~ .x == "convert_to_tele")
      )

      # efficiencies
      ## ip
      tc_e <- p$strategy_params$los_reduction |>
        purrr::discard(~ .x$type == "bads") |>
        names()
      shiny::updateSelectInput(session, "ip_am_e", choices = tc_e)
    })

    shiny::observeEvent(input$ip_am_a, {
      cf <- shiny::req(input$ip_am_a)
      p <- params()$strategy_params$admission_avoidance[[cf]]$interval

      shiny::updateTextInput(session, "ip_am_a_lo", value = p[[1]])
      shiny::updateTextInput(session, "ip_am_a_hi", value = p[[2]])
    })

    shiny::observeEvent(input$ip_am_tc_bads, {
      cf <- shiny::req(input$ip_am_tc_bads)
      p <- params()$strategy_params$los_reduction[[cf]]

      shiny::updateTextInput(session, "ip_am_tc_bads_br", value = p$baseline_target_rate)
      shiny::updateTextInput(session, "ip_am_tc_bads_split", value = p$op_dc_split)
      shiny::updateTextInput(session, "ip_am_tc_bads_lo", value = p$interval[[1]])
      shiny::updateTextInput(session, "ip_am_tc_bads_hi", value = p$interval[[2]])
    })

    shiny::observeEvent(input$ip_am_e, {
      cf <- shiny::req(input$ip_am_e)
      p <- params()$strategy_params$los_reduction[[cf]]$interval

      shiny::updateTextInput(session, "ip_am_e_lo", value = p[[1]])
      shiny::updateTextInput(session, "ip_am_e_hi", value = p[[2]])
    })

    shiny::observeEvent(input$op_am_a, {
      cf <- shiny::req(input$op_am_a)
      p <- params()$outpatient_factors[[cf]]

      shiny::updateTextInput(session, "op_am_a_lo_c1", value = p[["child_non-surgical"]]$interval[[1]])
      shiny::updateTextInput(session, "op_am_a_hi_c1", value = p[["child_non-surgical"]]$interval[[2]])

      shiny::updateTextInput(session, "op_am_a_lo_c2", value = p[["child_surgical"]]$interval[[1]])
      shiny::updateTextInput(session, "op_am_a_hi_c2", value = p[["child_surgical"]]$interval[[2]])

      shiny::updateTextInput(session, "op_am_a_lo_a1", value = p[["adult_non-surgical"]]$interval[[1]])
      shiny::updateTextInput(session, "op_am_a_hi_a1", value = p[["adult_non-surgical"]]$interval[[2]])

      shiny::updateTextInput(session, "op_am_a_lo_a2", value = p[["adult_surgical"]]$interval[[1]])
      shiny::updateTextInput(session, "op_am_a_hi_a2", value = p[["adult_surgical"]]$interval[[2]])
    })

    shiny::observeEvent(input$op_am_tc_tele, {
      cf <- shiny::req(input$op_am_tc_tele)
      p <- params()$outpatient_factors[[cf]]

      shiny::updateTextInput(session, "op_am_tc_tele_lo_c1", value = p[["child_non-surgical"]]$interval[[1]])
      shiny::updateTextInput(session, "op_am_tc_tele_hi_c1", value = p[["child_non-surgical"]]$interval[[2]])

      shiny::updateTextInput(session, "op_am_tc_tele_lo_c2", value = p[["child_surgical"]]$interval[[1]])
      shiny::updateTextInput(session, "op_am_tc_tele_hi_c2", value = p[["child_surgical"]]$interval[[2]])

      shiny::updateTextInput(session, "op_am_tc_tele_lo_a1", value = p[["adult_non-surgical"]]$interval[[1]])
      shiny::updateTextInput(session, "op_am_tc_tele_hi_a1", value = p[["adult_non-surgical"]]$interval[[2]])

      shiny::updateTextInput(session, "op_am_tc_tele_lo_a2", value = p[["adult_surgical"]]$interval[[1]])
      shiny::updateTextInput(session, "op_am_tc_tele_hi_a2", value = p[["adult_surgical"]]$interval[[2]])
    })

    shiny::observeEvent(input$aae_am_a, {
      cf <- shiny::req(input$aae_am_a)
      p <- params()$aae_factors[[cf]]

      shiny::updateTextInput(session, "aae_am_a_lo_c1", value = p[["child_ambulance"]]$interval[[1]])
      shiny::updateTextInput(session, "aae_am_a_hi_c1", value = p[["child_ambulance"]]$interval[[2]])

      shiny::updateTextInput(session, "aae_am_a_lo_c2", value = p[["child_walk-in"]]$interval[[1]])
      shiny::updateTextInput(session, "aae_am_a_hi_c2", value = p[["child_walk-in"]]$interval[[2]])

      shiny::updateTextInput(session, "aae_am_a_lo_a1", value = p[["adult_ambulance"]]$interval[[1]])
      shiny::updateTextInput(session, "aae_am_a_hi_a1", value = p[["adult_ambulance"]]$interval[[2]])

      shiny::updateTextInput(session, "aae_am_a_lo_a2", value = p[["adult_walk-in"]]$interval[[1]])
      shiny::updateTextInput(session, "aae_am_a_hi_a2", value = p[["adult_walk-in"]]$interval[[2]])
    })

    shiny::observeEvent(input$submit_model_run, {
      params <- shiny::req(params())

      shinyjs::disable("submit_model_run")
      shiny::updateActionButton(session, "submit_model_run", "Submitted...")

      job_name <- batch_add_job(params)

      status(paste("Submitted to batch:", job_name, "(Goto running models tab to view progress)"))
    })

    output$download_params <- shiny::downloadHandler(
      filename = function() {
        params <- shiny::req(params())
        c(ds, sc) %<-% params[c("input_data", "name")]
        glue::glue("{ds}_{sc}.json")
      },
      content = function(file) {
        params <- shiny::req(params())
        jsonlite::write_json(params, file, pretty = TRUE, auto_unbox = TRUE)
      }
    )

    output$status <- shiny::renderText({
      status()
    })
  })
}