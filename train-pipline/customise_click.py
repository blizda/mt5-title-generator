import click
import json


def CommandConfigFile(config_file_param_name):

    class CustomCommandClass(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            if config_file is not None:
                with open(config_file) as f:
                    data = json.load(f)
                    for param, value in ctx.params.items():
                        if value is None and param in data:
                            ctx.params[param] = data[param]

            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass