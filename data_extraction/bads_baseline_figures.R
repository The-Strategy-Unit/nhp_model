library(tidyverse)
library(DBI)

con <- dbConnect(
  odbc::odbc(),
  .connection_string = Sys.getenv("CONSTR"), timeout = 10
)

ip_bads_strategies <- tbl(
  con,
  dbplyr::in_schema("nhp_modelling", "strategies")
) |>
  filter(strategy %LIKE% "bads_%")

ip <- tbl(con, dbplyr::in_schema("nhp_modelling", "inpatients")) |>
  filter(FYEAR == 201819) |>
  inner_join(a, by = "EPIKEY") |>
  count(PROCODE3, CLASSPAT, strategy) |>
  collect() |>
  ungroup() |>
  mutate(
    type = ifelse(CLASSPAT == 1, "elective", "daycase"), .before = CLASSPAT
  ) |>
  select(-CLASSPAT)

op_procedures_op_or_dc <- tbl(con, "tbInpatientsProcedures") |>
  filter(OPORDER == 1, (
    OPCODE %LIKE% "H40[123]%" |
      OPCODE %LIKE% "H41[23]%" |
      OPCODE %LIKE% "Q16%" |
      OPCODE %LIKE% "Q17%" |
      OPCODE %LIKE% "Q03%"
  ))

op_procedures_op <- tbl(con, "tbInpatientsProcedures") |>
  filter(OPORDER == 1, (
    OPCODE %LIKE% "C10[16]%" |
      OPCODE %LIKE% "C111%" |
      OPCODE %LIKE% "C12[124568]%" |
      OPCODE %LIKE% "C222%" |
      OPCODE %LIKE% "C15[1245]%" |
      OPCODE %LIKE% "C39[1-9]%" |
      OPCODE %LIKE% "C432%" |
      OPCODE %LIKE% "C623%" |
      OPCODE %LIKE% "C664%" |
      OPCODE %LIKE% "S64[1-9]%" |
      OPCODE %LIKE% "S68[1-9]%" |
      OPCODE %LIKE% "S70[1-9]%" |
      OPCODE %LIKE% "W262%" |
      OPCODE %LIKE% "M293%" |
      OPCODE %LIKE% "M336%" |
      OPCODE %LIKE% "M275%" |
      OPCODE %LIKE% "M45[1-9]%" |
      OPCODE %LIKE% "N17[1-9]%" |
      OPCODE %LIKE% "L88[123]%" |
      OPCODE %LIKE% "L858%" |
      OPCODE %LIKE% "L86[29]%" |
      OPCODE %LIKE% "H52[34]%" |
      OPCODE %LIKE% "P27[389]%" |
      OPCODE %LIKE% "Q01[34]%" |
      OPCODE %LIKE% "Q02[1-9]%" |
      OPCODE %LIKE% "Q554%" |
      OPCODE %LIKE% "Q18[1-9]%" |
      OPCODE %LIKE% "Q20[2589]%" |
      OPCODE %LIKE% "V09[12]%"
  ))

op_count_op_or_dc <- tbl(
  con,
  dbplyr::in_schema("nhp_modelling", "inpatients")
) |>
  filter(ADMIMETH %LIKE% "1%") |>
  semi_join(op_procedures_op_or_dc, by = "EPIKEY") |>
  count(PROCODE3) |>
  collect()

op_count_op <- tbl(con, dbplyr::in_schema("nhp_modelling", "inpatients")) |>
  filter(ADMIMETH %LIKE% "1%") |>
  semi_join(op_procedures_op, by = "EPIKEY") |>
  count(PROCODE3) |>
  collect()

dbDisconnect(con)

bads_counts <- bind_rows(
  ip,
  op_count_op_or_dc |>
    mutate(type = "outpatients", strategy = "bads_outpatients_or_daycase"),
  op_count_op |>
    mutate(type = "outpatients", strategy = "bads_outpatients"),
)

bads_counts |>
  group_by(PROCODE3) |>
  count(type, strategy, wt = n) |>
  group_by(strategy, .add = TRUE) |>
  mutate(across(n, ~ .x / sum(.x))) |>
  ungroup() |>
  pivot_wider(names_from = type, values_from = n, values_fill = 0) |>
  mutate(r = case_when(
    strategy == "bads_outpatients" ~ outpatients,
    TRUE ~ 1 - elective
  )) |>
  select(-(daycase:outpatients)) |>
  filter(str_starts(PROCODE3, "R")) -> bads_results

bads_results |>
  pivot_wider(names_from = strategy, values_from = r) |>
  arrange(PROCODE3) |>
  print(n = 1000)

bads_results |>
  ggplot(aes(r, strategy, colour = strategy)) +
  geom_boxplot()
