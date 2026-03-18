library(tidyverse)

pairing64 <- list(
  c(1, 16), c(2, 15), c(3, 14), c(4, 13), c(5, 12), c(6, 11),  c(7, 10), c(8, 9)
)
pairing32 <- list(
  c(1, 16, 8, 9), c(2, 15, 7, 10), c(3, 14, 6, 11), c(4, 13, 5, 12)
)
pairing16 <- list(
  c(1, 16, 8, 9, 5, 12, 4, 13), c(2, 15, 7, 10, 6, 11, 3, 14)
)
pairing_semi <- list(
  c("s", "w"), c("e", "mw")
)

play <- function(v_team, v_wr) {
  sample(v_team, 1, prob = c(v_wr[1] * (1 - v_wr[2]), v_wr[2] * (1 - v_wr[1])))
}

win_rate <- read_csv("win_rate_2024.csv")

team <- read_csv("team.csv") %>%
  mutate(
    win_rate = apply(
      select(., team, team2),
      1,
      function(v) {
        if (!is.na(v[2])) {
          v[1] <- play(v, filter(win_rate, team %in% v) %>% pull(win_rate))
        }
        wr <- filter(win_rate, team == v[1]) %>% pull(win_rate)
        if (length(wr) == 0) wr <- NaN
        return(wr)
      }
    ) %>%
      unlist(),
    team2 = NULL
  )

res8_team <- sapply(
  split(team, team$region),
  function(region) {
    print(paste0("Region = ", unique(region$region)))

    res64 <- sapply(
      pairing64,
      function(v) play(v, filter(region, rank %in% v) %>% pull(win_rate))
    )
    print(res64)

    res32 <- sapply(
      lapply(pairing32, function(v) v[v %in% res64]),
      function(v) play(v, filter(region, rank %in% v) %>% pull(win_rate))
    )
    print(res32)

    res16 <- sapply(
      lapply(pairing16, function(v) v[v %in% res32]),
      function(v) play(v, filter(region, rank %in% v) %>% pull(win_rate))
    )
    print(res16)

    res8 <- play(res16, filter(region, rank %in% res16) %>% pull(win_rate))
    print(res8)
    return(filter(region, rank == res8) %>% pull(team))
  }
)

res_semi <- sapply(
  lapply(pairing_semi, function(v) res8_team[names(res8_team) %in% v]),
  function(v) play(v, filter(team, team %in% v) %>% pull(win_rate))
)
print(res_semi)

res_fin <- play(res_semi, filter(team, team %in% res_semi) %>% pull(win_rate))
print(res_fin)
