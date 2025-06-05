import click


class ConfigOptionGroup(click.Command):
    def parse_args(self, ctx, args):
        config_args = []
        other_args = []

        i = 0
        while i < len(args):
            arg = args[i]

            # Handle --config.key=value format
            if arg.startswith('--config.') and '=' in arg:
                key, value = arg.split('=', 1)
                config_args.extend([key, value])
                i += 1

            # Handle --config.key value format
            elif arg.startswith('--config.'):
                if i + 1 >= len(args):
                    raise click.BadParameter(f"Value required for {arg}")
                if args[i + 1].startswith('--'):
                    raise click.BadParameter(
                        f"Value required for {arg}, got another flag: {args[i + 1]}"
                    )
                config_args.extend([arg, args[i + 1]])
                i += 2

            # Handle other arguments
            else:
                other_args.append(arg)
                i += 1

        # Validate config args are properly paired
        if len(config_args) % 2 != 0:
            raise click.BadParameter("Malformed config override arguments")

        # Create a dictionary of config overrides
        config_dict = {}
        for i in range(0, len(config_args), 2):
            key = config_args[i].replace('--config.', '')
            value = config_args[i + 1]
            config_dict[key] = value

        # Store in context for access in the command
        ctx.obj = config_dict

        return super().parse_args(ctx, other_args)
