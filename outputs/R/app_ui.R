#' The application User-Interface
#'
#' @param request Internal parameter for `{shiny}`.
#'     DO NOT REMOVE.
#' @import bs4Dash
#' @noRd
app_ui <- function(request) {
  header <- dashboardHeader(title = "NHP Model Results")

  sidebar <- dashboardSidebar(
    sidebarMenu(
      id = "sidebarMenu",
      sidebarHeader("Model Run Selection"),
      mod_result_selection_ui("result_selection"),
      sidebarHeader("Principal Projection"),
      menuItem(
        text = "High Level",
        tabName = "tab_phl"
      ),
      menuItem(
        text = "Detailed",
        tabName = "tab_pd"
      ),
      menuItem(
        text = "Change Factors",
        tabName = "tab_pcf"
      ),
      sidebarHeader("Model Results"),
      menuItem(
        text = "Core Activity",
        tabName = "tab_mc"
      ),
      menuItem(
        text = "Results Distribution",
        tabName = "tab_md"
      )
    )
  )

  body <- dashboardBody(
    tabItems(
      tabItem(
        tabName = "tab_phl",
        mod_principal_high_level_ui("principal_high_level")
      ),
      tabItem(
        tabName = "tab_pd",
        mod_principal_detailed_ui("principal_detailed")
      ),
      tabItem(
        tabName = "tab_pcf",
        mod_principal_change_factor_effects_ui("principal_change_factor_effects")
      ),
      tabItem(
        tabName = "tab_mc",
        mod_model_core_activity_ui("model_core_activity")
      ),
      tabItem(
        tabName = "tab_md",
        mod_model_results_distribution_ui("model_results_distribution")
      )
    )
  )

  tagList(
    golem_add_external_resources(),
    dashboardPage(
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
  add_resource_path(
    "www", app_sys("app/www")
  )

  tags$head(
    favicon(),
    bundle_resources(
      path = app_sys("app/www"),
      app_title = "outputs"
    )
  )
}