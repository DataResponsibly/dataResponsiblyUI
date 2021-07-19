from django.conf.urls import url

from . import views

app_name = 'rankingfacts'
urlpatterns = [
    url(r'^$', views.base, name='base'),
    # urls for upload data and sample data
    url(r'^upload_data/$', views.upload_data, name='upload_data'),
    url(r'^sample_data/$', views.sample_data, name='sample_data'),
    # urls for parameters setting at without processing mode
    url(r'^data_process/$', views.data_process, name='data_process'),
    url(r'^json_processing_data/$', views.json_processing_data, name='json_processing_data'),
    url(r'^json_processing_hist/$', views.json_processing_hist, name='json_processing_hist'),
    url(r'^json_generate_ranking/$', views.json_generate_ranking, name='json_generate_ranking'),
    # urls for parameters setting at cleansing mode
    url(r'^norm_process/$', views.norm_process, name='norm_process'),
    url(r'^norm_json_processing_data/$', views.norm_json_processing_data, name='norm_json_processing_data'),
    url(r'^norm_json_processing_hist/$', views.norm_json_processing_hist, name='norm_json_processing_hist'),
    url(r'^norm_json_generate_ranking/$', views.norm_json_generate_ranking, name='norm_json_generate_ranking'),

    # urls for results page
    url(r'^nutrition_facts/$', views.nutrition_facts, name='nutrition_facts'),
    url(r'^json_scatter_score/$', views.json_scatter_score, name='json_scatter_score'),
    url(r'^json_piechart_data/$', views.json_piechart_data, name='json_piechart_data'),
]
