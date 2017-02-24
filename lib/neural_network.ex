defmodule NeuralNetwork do

  require Logger

  def time_of(function, args) do
    {time, result} = :timer.tc(function, args)
    Logger.debug "Time: #{time / 1.0e6}s"
    result
  end

  def sig(x) do
    1 / (1 + :math.exp(-x))
  end

  def sigd(x) do
    e = :math.exp(-x)
    e / :math.pow(1 + e, 2)
  end

  def dot(l1, l2) do
    Stream.zip(l1, l2)
    |> Enum.reduce(0, fn {x, y}, acc -> acc + x * y end)
  end

  def softmax(t) do
    {l, s} = Enum.map_reduce(t, 0, fn(x, acc) -> 
      e = :math.exp(x)
      {e, acc + e} 
    end)
    Enum.map(l, &(&1 / s))
  end

  def diff(a, b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} -> Enum.zip(x, y) 
      |> Enum.map(fn {m, n} -> abs(m - n) end) 
      |> Enum.sum end)
    |> Enum.sum
  end

  def max_index(l) do
    Enum.with_index(l, 1)
    |> Enum.max_by(&(elem(&1, 0)))
    |> elem(1)
  end

  def init(p, m, k) do
    :rand.seed(
      :exs64,
      {:erlang.phash2([node()]),
       :erlang.monotonic_time(),
       :erlang.unique_integer()
      })
    alpha = Enum.map(1..m, fn _  -> 
      Enum.map(1..p+1, fn _ -> 
        :rand.normal() / 100 
      end) 
    end)
    beta = Enum.map(1..k, fn _  -> 
      Enum.map(1..m+1, fn _ -> 
        :rand.normal() / 100 
      end) 
    end)
    {alpha, beta}
  end

  def forward(xs, alpha, beta) do
    zs = Enum.map(xs, fn x -> 
      [1 | Enum.map(alpha, fn a -> dot(a, x) |> sig end)]
    end)
    ts = Enum.map(zs, fn x ->
      Enum.map(beta, fn a -> dot(a, x) end)
    end)
    vs = Enum.map(ts, fn t -> softmax(t) end)
    {zs, ts, vs}
  end

  def backward(xs, ys, alpha, beta, garma, zs, ts, vs, p, m, k) do
    c1 = Enum.zip(ys, vs)
      |> Enum.map(fn {y, v} -> Enum.zip(y, v) 
        |> Enum.map(fn {e1, e2} -> (e1 - e2) * e2 end) end)
    c2 = Enum.zip([ys, vs, Enum.map(c1, fn x -> Enum.sum(x) end)])
      |> Enum.map(fn {y, v, c} -> Enum.zip(y, v)
        |> Enum.map(fn {e1, e2} -> (e1 - e2 - c) * e2 * 2 end) end)
    beta_new = Enum.map(0..k-1, fn x -> 
      Enum.map(0..m, fn y -> 
        Enum.at(Enum.at(beta, x), y) + 1 / garma * dot(Enum.map(c2, fn c -> Enum.at(c, x) end), Enum.map(zs, fn z -> Enum.at(z, y) end))
      end) 
    end)
    c3 = Enum.zip([c1, zs, ts])
      |> Enum.map(fn {c, z, t} -> 
        Enum.map(1..m, fn i -> 
          b1 = Enum.map(beta, fn e -> Enum.at(e, i) end)
          t1 = Enum.map(t, fn e -> :math.exp(e) end)
          t2 = dot(t1, b1) / Enum.sum(t1)
          b2 = Enum.map(b1, fn e -> e - t2 end)
          sigd(Enum.at(z, i)) * dot(c, b2) * 2
        end) 
      end)
    alpha_new = Enum.map(0..m-1, fn x ->
      Enum.map(0..p, fn y -> 
        Enum.at(Enum.at(alpha, x), y) + 1 / garma * dot(Enum.map(c3, fn c -> Enum.at(c, x) end), Enum.map(xs, fn x -> Enum.at(x, y) end))
      end)
    end)
    {alpha_new, beta_new}
  end

  def train({xs, ys, alpha, beta, garma, min_garma, max_diff, p, m, k}) do
    {zs, ts, vs} = forward(xs, alpha, beta)
    {alpha_new, beta_new} = backward(xs, ys, alpha, beta, garma, zs, ts, vs, p, m, k)
    d = diff(alpha, alpha_new) + diff(beta, beta_new)
    Logger.info "d"
    Logger.debug "#{inspect(d)}"
    case d < max_diff and garma > min_garma do
      false -> train({xs, ys, alpha_new, beta_new, garma+1, min_garma, max_diff, p, m, k})
      true -> {alpha_new, beta_new}
    end
  end

  def test(y0, y1) do
    {mistake, correct} =
      Enum.zip(y0, y1)
      |> Enum.reduce({0, 0}, fn {e1, e2}, {e, c} -> 
        case max_index(e1) == max_index(e2) do
          true -> {e, c+1}
          false -> {e+1, c}
        end
      end)
    mistake / (mistake + correct)
  end

  def simple_test() do
    p = 3
    m = 4
    k = 2
    xs = [
      [1,1.2,2.3,1.3],
      [1,3.2,-1,2.4],
      [1,2,2.5,0.3],
      [1,5,-3,4.1]
    ]
    {alpha0, beta0} = init(p, m, k)
    ys = [
      [1,0],
      [0,1],
      [1,0],
      [0,1]
    ]
    {alpha1, beta1} = train({xs, ys, alpha0, beta0, 1, 100, 1, p, m, k})
    ys_result = forward(xs, alpha1, beta1) |> elem(2)
    test(ys, ys_result)
  end

  def complex_test(m, garma, min_garma, max_diff) do
    {xs_train, ys_train} = load_data("train")
    {xs_test, ys_test} = load_data("test")
    p = length(Enum.at(xs_train, 0)) - 1
    k = length(Enum.at(ys_train, 0))
    {alpha0, beta0} = init(p, m, k)
    {alpha1, beta1} = time_of(&train(&1),[{xs_train, ys_train, alpha0, beta0, garma, min_garma, max_diff, p, m, k}])
    ys_train_result = forward(xs_train, alpha1, beta1) |> elem(2)
    ys_test_result = forward(xs_test, alpha1, beta1) |> elem(2)
    {test(ys_train, ys_train_result), test(ys_test, ys_test_result)}
  end

  defp load_data(s) do
    File.stream!("./data/" <> s) |> Enum.reduce({[], []}, fn l, {xs, ys} ->
      [h | t] = String.split(l, " ", trim: true) |> Enum.map(fn e -> Float.parse(e) |> elem(0) end)
      case h do
        2.0 -> {[[1 | t] | xs], [[1,0] | ys]}
        3.0 -> {[[1 | t] | xs], [[0,1] | ys]}
        _ -> raise "load " <> s <> " data error" 
      end
    end)
  end

end
