library(tidyverse)

get_kh03_file <- function() {
  url <- "https://raw.githubusercontent.com/nhs-r-community/demos-and-how-tos/main/kh03/kh03.Rds"
  file <- withr::local_tempfile(fileext = ".rds")
  download.file(url, file, quiet = TRUE)
  readRDS(file)
}

get_kh03_file <- function() {
  readRDS(file.path(Sys.getenv("USERPROFILE"), "Downloads", "kh03.Rds"))
}

kh03 <- get_kh03_file() |>
  # we are only interested, for now, in 2018/19
  filter(quarter |> str_detect("2018-19")) |>
  # use the overall occupancy rate to estimate available beds by specialty
  mutate(rate = occupied_total / available_total) |>
  unnest(by_specialty) |>
  mutate(available = occupied / rate, .before = occupied) |>
  # summarise rows to the year
  group_by(org_code, specialty_group, specialty_code) |>
  summarise(across(c(available, occupied), mean), .groups = "drop_last") |>
  # lump any specialty with < 1 occupied bed into the largest specialty
  arrange(desc(occupied)) |>
  mutate(
    across(specialty_code, ~ ifelse(occupied < 1, first(specialty_code), .x))
  ) |>
  # re-summarise the lumped data
  group_by(specialty_code, .add = TRUE) |>
  summarise(across(c(available, occupied), sum), .groups = "drop")

kh03_synthetic <- kh03 |>
  filter(specialty_group == "general_and_acute") |>
  group_by(org_code) |>
  summarise(across(where(is.numeric), sum)) |>
  filter(available |> between(600, 900)) |>
  semi_join(x = kh03, by = "org_code") |>
  complete(org_code, nesting(specialty_group, specialty_code)) |>
  mutate(across(where(is.numeric), replace_na, 0)) |>
  group_by(across(starts_with("specialty"))) |>
  summarise(across(where(is.numeric), mean), .groups = "drop_last") |>
  filter(occupied >= 1)

data_path <- "../nhp/nhp_model/data"
orgs <- dir(data_path) |>
  str_subset("^R") |>
  tibble(trust = _) |>
  mutate(org_code = str_split(trust, "_")) |>
  unnest(org_code)

kh03 |>
  inner_join(orgs, by = c("org_code")) |>
  select(-org_code) |>
  group_by(across(where(is.character))) |>
  summarise(across(where(is.numeric), sum), .groups = "drop") |>
  group_nest(trust) |>
  pwalk(\(trust, data) {
    write_csv(data, file.path(data_path, trust, "kh03.csv"))
  })


kh03 |>
  filter(specialty_code %in% c(190, 192)) |>
  complete(org_code, specialty_code) |>
  replace_na(list(occupied = 0)) |>
  ggplot(aes(specialty_code, occupied)) + geom_violin() + ggbeeswarm::geom_beeswarm()

kh03 |>
  filter(specialty_code %in% c(190, 192)) |>
  arrange(desc(occupied)) |>
  select(-specialty_group, -available) |>
  pivot_wider(names_from = specialty_code, values_from = occupied) |>
  mutate(across(-org_code, replace_na, 0)) |>
  mutate(total = `190` + `192`) |>
  arrange(desc(total))
