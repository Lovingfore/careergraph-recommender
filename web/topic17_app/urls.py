"""Web 页面、JSON 服务与 SVG 图谱资源的 URL 映射。"""

from django.urls import path

from .views import (
    bipartite_graph,
    career_path_api,
    forecast_api,
    index,
    model_info_api,
    occupations_api,
    recommendation_api,
    resume_upload_api,
    skill_profile_api,
    skill_gap_api,
    summary_api,
    transition_graph,
)


urlpatterns = [
    # 首页：由 Django 注入首屏摘要，再由浏览器端脚本加载交互数据。
    path("", index, name="index"),

    # 系统摘要、模型元数据和职位字典均为只读查询接口。
    path("api/summary/", summary_api, name="summary_api"),
    path("api/model-info/", model_info_api, name="model_info_api"),
    path("api/occupations/", occupations_api, name="occupations_api"),

    # 简历即时分析，以及已有用户的推荐、技能画像、缺口、路径和趋势预测接口。
    path("api/resume-upload/", resume_upload_api, name="resume_upload_api"),
    path("api/recommend/", recommendation_api, name="recommendation_api"),
    # 技能画像接口返回每项技能的最新水平，是已有用户雷达图的后端数据源。
    path("api/skill-profile/", skill_profile_api, name="skill_profile_api"),
    path("api/skill-gap/", skill_gap_api, name="skill_gap_api"),
    path("api/career-path/", career_path_api, name="career_path_api"),
    path("api/forecast/", forecast_api, name="forecast_api"),

    # 图谱资源：二部图读取已生成文件，转移图根据当前数据动态生成 SVG。
    path("bipartite-graph.svg", bipartite_graph, name="bipartite_graph"),
    path("transition-graph.svg", transition_graph, name="transition_graph"),
]
