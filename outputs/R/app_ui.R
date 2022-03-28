#' The application User-Interface
#'
#' @param request Internal parameter for `{shiny}`.
#'     DO NOT REMOVE.
#' @import bs4Dash
#' @noRd
app_ui <- function(request) {
  header <- dashboardHeader(title = "Basic dashboard")

  sidebar <- dashboardSidebar(
    sidebarMenu(
      id = "sidebarMenu",
      sidebarHeader("Principal Projection"),
      menuItem(
        text = "High Level",
        tabName = "tab_phl"
      ),
      menuItem(
        text = "Detailed",
        tabName = "tab_pd"
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
golem_add_external_resources <- function(){

  add_resource_path(
    'www', app_sys('app/www')
  )

  tags$head(
    favicon(),
    bundle_resources(
      path = app_sys('app/www'),
      app_title = 'outputs'
    )
  )
}
