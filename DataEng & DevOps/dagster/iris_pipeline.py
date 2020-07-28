import dagstermill as dm
from dagster import InputDefinition, pipeline
from dagster.utils import script_relative_path
import os
from six.moves.urllib.request import urlretrieve
from dagster import Field, OutputDefinition, String, solid
from dagster import Int

k_means_iris = dm.define_dagstermill_solid(
    'k_means_iris',
    script_relative_path('iris-kmeans.ipynb'),
    input_defs=[InputDefinition('path', str, description='Local path to the Iris dataset')],
    output_defs=[OutputDefinition(Int, name="clusters_used", description="Number of clusters used")],
    config_schema=Field(Int, default_value=3, is_required=False, description='The number of clusters to find'
    ),
)

@solid(
    name='download_file',
    config_schema={
        'url': Field(String, description='The URL from which to download the file'),
        'path': Field(String, description='The path to which to download the file'),
    },
    output_defs=[
        OutputDefinition(
            String, name='path', description='The path to which the file was downloaded'
        )
    ],
    description=(
        'A simple utility solid that downloads a file from a URL to a path using '
        'urllib.urlretrieve'
    ),
)
def download_file(context):
    urlretrieve(context.solid_config['url'], context.solid_config['path'])
    return os.path.abspath(context.solid_config['path'])
    
@pipeline
def iris_pipeline():
    k_means_iris(download_file())
    
    