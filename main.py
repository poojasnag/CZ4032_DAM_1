from absl import app
from absl import flags
from CrossValM2 import CrossValidationM2


FLAGS = flags.FLAGS

flags.DEFINE_string('filename', 'iris', 'Path to data file.')
# flags.DEFINE_string('scheme_path', 'datasets/iris.names', 'Path to scheme file.')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_float('minsup', 0.01, 'Minimum support level')
flags.DEFINE_float('minconf', 0.5, 'Minimum confidence level')


def main(argv):

    test_data = [
        ['red', 25.6, 56, 1],
        ['green', 33.3, 1, 1],
        ['green', 2.5, 23, 0],
        ['blue', 67.2, 111, 1],
        ['red', 29.0, 34, 0],
        ['yellow', 99.5, 78, 1],
        ['yellow', 10.2, 23, 1],
        ['yellow', 9.9, 30, 0],
        ['blue', 67.0, 47, 0],
        ['red', 41.8, 99, 1]
    ]

    if FLAGS.debug:
        print("Non-flag arguments:", argv)
    # data_path = f"./datasets/{FLAGS.filename}.data"
    # scheme_path = f"./datasets/{FLAGS.filename}.names"

    test_attribute = ['color', 'average', 'age', 'class']
    test_value_type = ['categorical', 'numerical', 'numerical', 'label']
    
    print("data_path:", data_path)
    print("scheme_path:", scheme_path)
    minsup = FLAGS.minsup
    minconf = FLAGS.minconf

    validation = CrossValidationM2(data_path, scheme_path, minsup, minconf)
    validation.cross_validation()


if __name__ =="__main__":
    app.run(main)
