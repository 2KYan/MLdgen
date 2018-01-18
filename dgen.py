import argparse
import dgm.loader


def build_parser():
    parser = argparse.ArgumentParser(description="ML based data projection")
    parser.add_argument('-f', default="defaults.yaml", help='help')
    parser.add_argument('-cfg', default='override configurations')
    return parser

def run(args):
    ldr = dgm.loader.Loader(args)
    if ldr.load():
        feature_data, label_data, feature_columns = ldr.pd_filter()
        model = ldr.build(feature_columns)

        if model is not None:
            model.fit(feature_data, label_data)
            model.validate(feature_data, label_data)
            model.summary(feature_columns)
            model.test(feature_data, label_data)


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)