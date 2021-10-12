from absl import app
from absl import flags
from CrossValM2 import CrossValidationM2


FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', 'datasets/iris.data', 'Path to data file.')
flags.DEFINE_string('scheme_path', 'datasets/iris.names', 'Path to scheme file.')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_float('minsup', 0.01, 'Minimum support level')
flags.DEFINE_float('minconf', 0.5, 'Minimum confidence level')


def main(argv):
    if FLAGS.debug:
        print("Non-flag arguments:", argv)
    data_path = FLAGS.data_path
    scheme_path = FLAGS.scheme_path
    minsup = FLAGS.minsup
    minconf = FLAGS.minconf

    validation = CrossValidationM2(data_path, scheme_path, minsup, minconf)
    validation.cross_validation()


if __name__ =="__main__":
    app.run(main)
