from django.urls import path

from .views import (
    bipartite_graph,
    career_path_api,
    forecast_api,
    index,
    recommendation_api,
    skill_gap_api,
    summary_api,
)


urlpatterns = [
    path("", index, name="index"),
    path("api/summary/", summary_api, name="summary_api"),
    path("api/recommend/", recommendation_api, name="recommendation_api"),
    path("api/skill-gap/", skill_gap_api, name="skill_gap_api"),
    path("api/career-path/", career_path_api, name="career_path_api"),
    path("api/forecast/", forecast_api, name="forecast_api"),
    path("bipartite-graph.svg", bipartite_graph, name="bipartite_graph"),
]
