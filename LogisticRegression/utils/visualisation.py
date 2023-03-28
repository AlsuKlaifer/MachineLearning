import plotly.graph_objects as go


class Visualisation:

    @staticmethod
    def visualise_metrics(epochs: list, metrics: list, plot_title='', y_title=''):
        text = [y_title] * len(epochs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs,
                                 y=metrics,
                                 mode='lines+markers',
                                 hovertext=text,
                                 name='metrics'))
        fig.update_layout(title=plot_title,
                          xaxis_title='Numbers of training iteration',
                          yaxis_title=y_title)
        fig.update_traces(hoverinfo='all',
                          hovertemplate='Epoch: %{x}<br>%{hovertext}: %{y}')
        fig.show()
