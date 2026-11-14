import json
import random
import numpy as np


def init_states():
    settings = dict(
        width = 375,
        height = 667,
        platformWidth = 65,
        platformHeight = 20,

        gravity = 0.33,
        drag = 0.3,
        bounceVelocity = -12.5,

        minPlatformSpace = 15,
        maxPlatformSpace = 20,
        keydown = False,
        score = 0,
    )
    settings['platformStart'] = settings['height'] - 50

    platforms = [dict(x=settings['width'] / 2 - settings['platformWidth'] / 2,
                      y=settings['platformStart'])]
    y = settings['platformStart']
    while (y > 0):
        y -= settings['platformHeight'] + \
             np.random.randint(settings['minPlatformSpace'],
                               settings['maxPlatformSpace'])

        while True:
            x = np.random.uniform(25, settings['width'] \
                                  - 25 - settings['platformWidth'])
            if not ((y > settings['height'] / 2) and
                    (x > settings['width'] / 2
                     - settings['platformWidth'] * 1.5) and
                    (x < settings['width'] / 2
                     + settings['platformWidth'] / 2)):
                break
        platforms.append(dict(x=x, y=y))

    doodle = dict(
      width=40,
      height=60,
      x=settings['width'] / 2 - 20,
      y=settings['platformStart'] - 60,
      dx=0,
      dy=0,
      playerDir=0,
      prevDoodleY=settings['platformStart'] - 60,
    )

    return doodle, platforms, settings

doodle, platforms, settings = init_states()

def restart():
    doodle, platforms, settings = init_states()
    return doodle, platforms, settings


def loop(doodle, platforms, settings):
    doodle['dy'] += settings['gravity']

    if (doodle['y'] < settings['height'] / 2 and doodle['dy'] < 0):
        for i, _ in enumerate(platforms):
            platforms[i]['y'] -= doodle['dy']

        while (platforms[-1]['y'] > 0):
            platforms.append(dict(
                x=np.random.uniform(25,
                                    settings['width'] - 25
                                    - settings['platformWidth']),
                y=np.random.uniform(platforms[-1]['y'] -
                                    (settings['platformHeight'] +
                                     np.random.uniform(
                                         settings['minPlatformSpace'],
                                         settings['maxPlatformSpace']))
                                    )
                                )
                            )

            settings['minPlatformSpace'] = min(settings['minPlatformSpace']
                                               + 0.5,
                                               settings['height'] / 2 - 0.5)
            settings['maxPlatformSpace'] = min(settings['maxPlatformSpace']
                                               + 0.5,
                                               settings['height'] / 2)
    else:
        doodle['y'] += doodle['dy']

    if not settings['keydown']:
        if (doodle['playerDir'] < 0):
            doodle['dx'] += settings['drag'];
            if (doodle['dx'] > 0):
                doodle['dx'] = 0
                doodle['playerDir'] = 0
        elif (doodle['playerDir'] > 0):
            doodle['dx'] -= settings['drag']

            if (doodle['dx'] < 0):
                doodle['dx'] = 0
                doodle['playerDir'] = 0

    doodle['x'] += doodle['dx']

    if (doodle['x'] + doodle['width'] < 0):
        doodle['x'] = settings['width']
    elif (doodle['x'] > settings['width']):
        doodle['x'] = -doodle['width']

    for platform in platforms:
        if (
          (doodle['dy'] > 0) and
          (doodle['prevDoodleY'] + doodle['height'] <= platform['y']) and
          (doodle['x'] < platform['x'] + settings['platformWidth']) and
          (doodle['x'] + doodle['width'] > platform['x']) and
          (doodle['y'] < platform['y'] + settings['platformHeight']) and
          (doodle['y'] + doodle['height'] > platform['y'])
        ):
            doodle['y'] = platform['y'] - doodle['height']
            doodle['dy'] = settings['bounceVelocity']

    doodle['prevDoodleY'] = doodle['y']
    platforms_cleared = len(platforms)
    platforms = list(filter(lambda platform: platform['y'] < settings['height'],
                            platforms))
    platforms_cleared -= len(platforms)
    settings['score'] += platforms_cleared

    if doodle['y'] > settings['height'] + doodle['height']:
        return -1
    return 0

actionMap = {0: 37, # движение влево
             1: 39, # движение вправо
             2: -1} # ожидание

def apply_action(doodle, platforms, settings, action):
    action = np.clip(action, -1.0, 1.0)

    if abs(action) < 0.05:
        settings['keydown'] = False
        return

    settings['keydown'] = True
    settings['playerDir'] = -1 if action < 0 else 1
    doodle['dx'] = action * 3.0
        
        # агент видел значения в этих пикселях
sensor_web = np.meshgrid(np.arange(-settings['width']*2//3,
                                   +settings['width']*2//3, 50),
                         np.arange(-settings['height']*2//3,
                                   +settings['height']*2//3, 75))
sensor_web = np.concatenate([sensor_web[0].flatten()[:, None],
                             sensor_web[1].flatten()[:, None]], axis=1)





def get_features(doodle, platforms, settings, prev_action_val):
    player_width = doodle['width']
    player_height = doodle['height']
    screen_width = settings['width']

    platform_width = settings['platformWidth']

    d_center_x = doodle['x'] + player_width / 2
    d_center_y = doodle['y'] + player_height / 2
    d_feet_y = doodle['y'] + player_height

    points = np.array([
        (p['x'] + platform_width / 2, p['y'])
        for p in platforms
    ]).reshape(-1, 2)

    norm_x = (d_center_x % screen_width) / screen_width
    norm_y = doodle['y'] / settings['height']
    norm_dx = doodle['dx'] / 5.0
    norm_dy = doodle['dy'] / 15.0

    def get_wrapped_dist_x(target_x, current_x, width):
        dx = target_x - current_x

        if dx > width / 2:
            dx -= width
        elif dx < -width / 2:
            dx += width

        return dx / width

    # Минимальное горизонтальное перемещение
    # для пересечения персонажа с платформой.
    # Учитываем переход через края экрана.
    def horizontal_gap(platform_left):
        player_left = doodle['x']
        player_right = player_left + player_width

        gaps = []

        for shift in (-screen_width, 0, screen_width):
            left = platform_left + shift
            right = left + platform_width

            if player_right < left:
                gap = left - player_right
            elif player_left > right:
                gap = right - player_left
            else:
                gap = 0.0

            gaps.append(gap)

        return min(gaps, key=abs)

    # Время до пересечения высоты платформы
    # при движении вниз.
    def time_to_landing(platform_y):
        dy = doodle['dy']
        gravity = settings['gravity']

        delta_y = platform_y - d_feet_y

        b = dy + gravity / 2
        discriminant = b**2 + 2 * gravity * delta_y

        if discriminant < 0:
            return None

        t = (-b + np.sqrt(discriminant)) / gravity

        if t <= 0 or dy + gravity * t <= 0:
            return None

        return t

    # Платформы снизу
    platforms_below = points[
        points[:, 1] >= d_center_y
    ]

    sorted_below = (
        platforms_below[
            np.argsort(platforms_below[:, 1])
        ] if len(platforms_below) > 0 else []
    )

    def get_below_feat(arr, idx):
        if len(arr) > idx:
            return (
                get_wrapped_dist_x(
                    arr[idx][0],
                    d_center_x,
                    screen_width
                ),
                (arr[idx][1] - d_feet_y) / settings['height']
            )

        return 0.0, 1.0

    dist_x_b1, dist_y_b1 = get_below_feat(sorted_below, 0)
    dist_x_b2, dist_y_b2 = get_below_feat(sorted_below, 1)

    # Кандидаты для будущего приземления
    MAX_JUMP = 230.0

    platforms_above = points[
        (points[:, 1] < d_feet_y) &
        ((d_feet_y - points[:, 1]) <= MAX_JUMP)
    ]

    landing_candidates = []

    for platform in platforms_above:
        t = time_to_landing(platform[1])

        if t is None:
            continue

        platform_left = platform[0] - platform_width / 2

        gap = horizontal_gap(platform_left)

        max_horizontal_distance = 3.0 * t

        reachability = (
            abs(gap) / max(max_horizontal_distance, 1.0)
        )

        reachability = min(reachability, 2.0) / 2.0

        landing_candidates.append((
            gap,
            t,
            reachability,
            platform[1]
        ))

    # Сортируем по времени до пересечения высоты
    landing_candidates.sort(key=lambda item: item[1])

    def get_above_feat(idx):
        if len(landing_candidates) > idx:
            gap, t, reachability, _ = landing_candidates[idx]

            return (
                gap / screen_width,
                min(t / 75.0, 1.0),
                reachability
            )

        return 0.0, 1.0, 1.0

    gap_a1, time_a1, reach_a1 = get_above_feat(0)
    gap_a2, time_a2, reach_a2 = get_above_feat(1)
    gap_a3, time_a3, reach_a3 = get_above_feat(2)
    gap_a4, time_a4, reach_a4 = get_above_feat(3)

    return np.array([
        # Персонаж: 4
        norm_x, norm_y, norm_dx, norm_dy,

        # Нижние платформы: 4
        dist_x_b1, dist_y_b1,
        dist_x_b2, dist_y_b2,

        # Кандидаты для приземления: 12
        gap_a1, time_a1, reach_a1,
        gap_a2, time_a2, reach_a2,
        gap_a3, time_a3, reach_a3,
        gap_a4, time_a4, reach_a4,

        # Предыдущее действие и bias: 2
        prev_action_val, 1.0
    ])



# ВНИМАНИЕ: Обязательно обновите размер входного слоя в get_one!
def get_one(h1=16, h2=8):
    W1 = np.random.normal(size=(22, h1))
    W2 = np.random.normal(size=(h1, h2))
    W3 = np.random.normal(size=(h2, 1))

    return W1, W2, W3


def getAction(doodle, platforms, settings, weights, prev_action_val):
    W1, W2, W3 = weights
    x = get_features(doodle, platforms, settings, prev_action_val)

    h1 = np.maximum(0, x.dot(W1))
    h2 = np.maximum(0, h1.dot(W2))
    output = h2.dot(W3)

    return float(np.tanh(output[0]))

def get_score(weights, patience=150, return_actions=False):
    doodle, platforms, settings = restart()
    patience_left = patience
    total_scroll = 0.0
    actions, xcoords = [], []

    prev_action_val = 0.0
    action = getAction(doodle, platforms, settings, weights, prev_action_val)

    # УДАЛИЛИ JITTER_PENALTY ПОЛНОСТЬЮ!

    for _ in range(int(5e4)):
        y_before = platforms[0]['y'] if len(platforms) > 0 else 0

        if loop(doodle, platforms, settings) == -1:
            break

        y_after = platforms[0]['y'] if len(platforms) > 0 else 0
        scroll = y_after - y_before

        if scroll > 0 and scroll < 100:
            total_scroll += scroll
            patience_left = patience

        action = getAction(doodle, platforms, settings, weights, prev_action_val)
        prev_action_val = action

        apply_action(doodle, platforms, settings, action)

        actions.append(action)
        xcoords.append(doodle['x'])

        patience_left -= 1
        if patience_left < 0:
            break

    score = total_scroll / 50.0
    # Теперь выживание и набор высоты — единственный критерий!

    if return_actions:
        return actions, xcoords, score
    return score



def mutate(weights, mutation_rate=0.02, mutation_prob=0.1):
    # Теперь мы распаковываем ТРИ матрицы
    W1, W2, W3 = weights

    # Вспомогательная функция, чтобы не писать одно и то же три раза
    def mutate_layer(layer):
        dW = np.random.normal(size=layer.shape)
        mask = np.random.random(size=layer.shape) < mutation_prob
        return layer + dW * mask * mutation_rate

    # Мутируем и возвращаем все три слоя
    return mutate_layer(W1), mutate_layer(W2), mutate_layer(W3)


def crossover(w1, w2):
    # Распаковываем матрицы обоих родителей
    W1_a, W2_a, W3_a = w1
    W1_b, W2_b, W3_b = w2

    # Вспомогательная функция для случайного обмена генами (весами)
    def cross_matrices(m1, m2):
        mask = np.random.random(size=m1.shape) < 0.5
        return np.where(mask, m1, m2)

    # Скрещиваем каждый слой по отдельности
    new_W1 = cross_matrices(W1_a, W1_b)
    new_W2 = cross_matrices(W2_a, W2_b)
    new_W3 = cross_matrices(W3_a, W3_b)

    return new_W1, new_W2, new_W3


def generate_random(population, size, current_mut_rate=0.02):
    new_population = []
    for _ in range(size):
        if np.random.random() < 0.5:
            new_population.append(get_one())
        else:
            new_population.append(mutate(population[0], mutation_rate=current_mut_rate))
    return new_population


def selection(population, scores, topK=4):
    scores = np.array(scores, dtype=float)
    scores /= scores.sum()

    elitismTopK = np.argsort(scores)[::-1][:topK]

    new_population = [
        tuple(layer.copy() for layer in population[i])
        for i in elitismTopK
    ]

    return new_population


def breed(population, scores, nChilds=10, current_mut_rate=0.02):
    scores = np.array(scores)*1.
    scores /= scores.sum()
    parents = np.random.choice(len(scores),
                               p=scores,
                               size=(nChilds, 2))

    new_population = []
    for parentA, parentB in parents:
        new_population.append(mutate(crossover(population[parentA],
                                               population[parentB]),
                                     mutation_rate=current_mut_rate))

    return new_population


# Округление весов до 3 знаков для сжатия финального js файла
def factorize(population, factor=3):
    for i, p in enumerate(population):
        population[i] = tuple([np.array([[int(10**factor*w)/10**factor
                                          for w in W]
                                         for W in pp])
                               for pp in p])
    return population


def get_new_population(population, scores, topK=4, randomNum=10, current_mut_rate=0.02):
    return (
        selection(population, scores, topK) +
        breed(
            population,
            scores,
            nChilds=max(0, len(population) - randomNum - topK),
            current_mut_rate=current_mut_rate
        ) +
        generate_random(population, randomNum, current_mut_rate)
    )
    
    
def get_scores(population, patience=100, pbar=None):
    scores = []
    for weights in population:
        scores.append(get_score(weights, patience))
        # Сдвигаем ползунок прогресса после каждой сыгранной игры
        if pbar is not None:
            pbar.update(1)
    return scores

import json

def save_thingey(weights, score):
    W1, W2, W3 = weights
    filename = f'doodlejump_weights_{score:.1f}.js'
    with open(filename, 'w') as f:
        # Сохраняем все три слоя в константы JS
        f.write(f"const W1 = {json.dumps(W1.tolist())};\n")
        f.write(f"const W2 = {json.dumps(W2.tolist())};\n")
        f.write(f"const W3 = {json.dumps(W3.tolist())};\n")
    print(f"Файл {filename} успешно сохранен!")
    
    
    
    
from tqdm.auto import tqdm
import concurrent.futures

POPULATION_SIZE = 64
RANDOM_SIZE = 20
ELITE_SIZE = 4
NUM_GENERATIONS = 80
NUM_REPEATS = 3
NUM_RESTARTS = 5

best_thingey = None
best_score = 0


def evaluate_agent(weights):
    return get_score(weights, patience=150)


def main():
    global best_thingey, best_score

    # Создаём пул процессов один раз на всё обучение
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:

        for n_restart in range(NUM_RESTARTS):
            print('=' * 50)
            print('Старт перезапуска №%d' % (n_restart + 1))
            print('=' * 50)

            population = [get_one() for _ in range(POPULATION_SIZE)]
            stagnation_counter = 0
            local_best_score = 0

            for generation in range(NUM_GENERATIONS):
                scores = np.zeros(POPULATION_SIZE)

                for _ in range(NUM_REPEATS):
                    results = list(tqdm(
                        executor.map(evaluate_agent, population),
                        total=POPULATION_SIZE,
                        desc=f'Поколение {generation}',
                        leave=False
                    ))

                    scores += np.array(results)

                scores /= NUM_REPEATS

                bscore = max(scores)
                mean_score = np.mean(scores)

                dynamic_mut_rate = 0.1 * (0.95 ** generation)

                print(
                    f'Итоги {generation} | '
                    f'Макс счет: {bscore:.1f} платформ | '
                    f'Средний: {mean_score:.1f} | '
                    f'Мутация: {dynamic_mut_rate:.3f}'
                )

                # Сохраняем именно лучшую особь текущего поколения
                best_idx = np.argmax(scores)

                if bscore > best_score:
                    best_score = bscore
                    best_thingey = [
                        layer.copy()
                        for layer in population[best_idx]
                    ]

                    print(
                        f'>>> НОВЫЙ ГЛОБАЛЬНЫЙ РЕКОРД! '
                        f'Счет: {best_score:.1f} <<<'
                    )

                    if bscore > 10.0:
                        save_thingey(best_thingey, best_score)

                if bscore > local_best_score:
                    local_best_score = bscore
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                selection_scores = scores ** 3

                population = get_new_population(
                    population,
                    selection_scores,
                    ELITE_SIZE,
                    RANDOM_SIZE,
                    dynamic_mut_rate
                )

                if stagnation_counter >= 40:
                    print('Эволюция застряла. Переходим к следующему рестарту!')
                    break

    if best_thingey is not None and best_score > 10.0:
        save_thingey(best_thingey, best_score)


if __name__ == "__main__":
    main()
