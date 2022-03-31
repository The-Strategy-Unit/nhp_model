#' The application User-Interface
#'
#' @param request Internal parameter for `{shiny}`.
#'     DO NOT REMOVE.
#' @noRd
app_ui <- function(request) {
  header <- bs4Dash::dashboardHeader(title = "NHP Model Results")

  sidebar <- bs4Dash::dashboardSidebar(
    bs4Dash::sidebarMenu(
      id = "sidebarMenu",
      bs4Dash::sidebarHeader("Model Run Selection"),
      mod_result_selection_ui("result_selection"),
      bs4Dash::sidebarHeader("Principal Projection"),
      bs4Dash::menuItem(
        text = "High Level",
        tabName = "tab_phl"
      ),
      bs4Dash::menuItem(
        text = "Detailed",
        tabName = "tab_pd"
      ),
      bs4Dash::menuItem(
        text = "Change Factors",
        tabName = "tab_pcf"
      ),
      bs4Dash::sidebarHeader("Model Results"),
      bs4Dash::menuItem(
        text = "Core Activity",
        tabName = "tab_mc"
      ),
      bs4Dash::menuItem(
        text = "Results Distribution",
        tabName = "tab_md"
      )
    )
  )

  body <- bs4Dash::dashboardBody(
    bs4Dash::tabItems(
      bs4Dash::tabItem(
        tabName = "tab_phl",
        mod_principal_high_level_ui("principal_high_level")
      ),
      bs4Dash::tabItem(
        tabName = "tab_pd",
        mod_principal_detailed_ui("principal_detailed")
      ),
      bs4Dash::tabItem(
        tabName = "tab_pcf",
        mod_principal_change_factor_effects_ui("principal_change_factor_effects")
      ),
      bs4Dash::tabItem(
        tabName = "tab_mc",
        mod_model_core_activity_ui("model_core_activity")
      ),
      bs4Dash::tabItem(
        tabName = "tab_md",
        mod_model_results_distribution_ui("model_results_distribution")
      )
    )
  )

  shiny::tagList(
    golem_add_external_resources(),
    bs4Dash::dashboardPage(
      header,
      sidebar,
      body
    )
  )
}
#' Add external Resources to the Application
#'
#' This function is internally used to add external
#' resources inside the Shiny application.
#'
#' @importFrom golem add_resource_path activate_js favicon bundle_resources
#' @noRd
golem_add_external_resources <- function() {
  golem::add_resource_path(
    "www", app_sys("app/www")
  )

  shiny::tags$head(
    golem::favicon(),
    golem::bundle_resources(
      path = app_sys("app/www"),
      app_title = "outputs"
    )
  )
}