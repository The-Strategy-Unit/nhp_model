#' Demographic Factors
#'
#' @param local_authorities a character vector listing the local authorities to select (use GSS local authority codes)
#' @param variant_probalities: the probability to select each of the variants, should be a named vector where the names
#'    correspond to the possible variants
#' @param start_year the start year to use as the numerator for the factor (defaults to minimum year in data)
#' @param end_year the end year to use as the denomintar for the factor (defaults to the maximum year in data)
#'
#' @examples
#' demographic_factors("E08000031", "principal")
#' demographic_factors("E08000031", "high migration")
#'
#' @export
demographic_factors <- function(local_authorities, variant_probabilities, start_year, end_year) {
  d <- system.file("data", "demographic_factors.csv", package = "nhpmodel") |>
    read_csv(col_types = paste(c(rep("c", 5), rep("d", 26)), collapse = ""),
             lazy = FALSE) |>
    filter(.data$age != "all") |>
    mutate(across(.data$age, compose(partial(pmin, 90), as.numeric))) |>
    pivot_longer(matches("\\d{4}"), names_repair = "minimal", names_to = "year", values_to = "pop") |>
    mutate(across(.data$year, as.numeric))

  if (missing(start_year)) {
    start_year <- min(d$year)
  }

  if (missing(end_year)) {
    end_year <- max(d$year)
  }

  function() {
    variant <- sample(names(variant_probabilities), 1, prob = variant_probabilities)

    structure(
      d |>
        filter(.data$code %in% .env$local_authorities, .data$variant == .env$variant) |>
        group_by(.data$sex, .data$age) |>
        mutate(a = .data$pop * (.data$year == .env$end_year),
               b = .data$pop * (.data$year == .env$start_year)) |>
        summarise(factor = sum(.data$a) / sum(.data$b), .groups = "drop") |>
        # prep for joining back to data
        rename(admiage = age) |>
        mutate(across(sex, ~ifelse(.x == "male", "1", "2"))),
      selected_variant = variant
    )
  }
}
